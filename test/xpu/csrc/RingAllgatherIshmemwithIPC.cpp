// RingAllgatherIshmemwithIPC.cpp
//
// Flat-ring allgather that picks the transport PER HOP from the hwloc-backed
// topology check (see Topology.hpp): the push to the right neighbour uses
//   - a DIRECT peer load/store (PCIe / Xe-Link, via an ISHMEM IPC pointer
//     obtained with ishmem_ptr) when the right neighbour is reachable over
//     PCIe, or
//   - an ISHMEM RDMA put (ishmemx_putmem_nbi_work_group) when the right
//     neighbour is only reachable through the NIC (e.g. a different node).
//
// Everything else mirrors RingAllgatherIshmem.cpp: a SINGLE on-device kernel
// runs the whole world_size-1 hop ring; the shard/slot is split into `num_wg`
// contiguous byte-slices and each work-group runs an independent pipeline over
// its slice (waits only on the SAME wg of the LEFT peer, signals only the SAME
// wg of the RIGHT peer, so no cross-wg wait and no deadlock). Cross-rank step
// ordering uses per-step signal pads with a strictly-increasing tag.
//
// Transport decision (Topology::transport_to):
//   same switch / same numa / same node  -> PCIe  (default) -> direct copy
//   different node                        -> NIC            -> ISHMEM put
// The PCIe path still requires the peer's symmetric buffer to be directly
// accessible; we get that pointer with ishmem_ptr(symm, right). If ishmem_ptr
// returns null (peer not IPC-mappable) we fall back to the NIC path for safety.
//
// The wait side must MATCH how the LEFT peer pushed to us (data and flag travel
// the same path per hop, and their ordering only holds under that path's
// semantics):
//   - LEFT on NIC  -> ishmem_uint64_wait_until (observes NIC-delivered signal)
//   - LEFT on PCIe -> system-scope acquire fence + plain load spin (pairs with
//                     the left peer's release fence after its direct store)
//
// Registered op: symm_mem::ring_allgather_ishmem_ipc

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <comm/SYCLHelpers.h>
#include <ishmem.h>
#include <ishmemx.h>
#include <mpi.h>

#include "Topology.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace {

struct RingState {
  std::mutex mutex;
  bool initialized = false;
  void* symm = nullptr;
  size_t symm_bytes = 0;
  uint64_t* pad = nullptr;
  int pad_slots = 0;
  uint64_t iteration = 0;
  bool topo_ready = false;
  Topology topo;
};

RingState& get_state() {
  static RingState state;
  return state;
}

bool env_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

bool debug_enabled() {
  return env_enabled("RING_ALLGATHER_IPC_DEBUG");
}

void debug_log(int64_t rank, const char* msg) {
  if (debug_enabled()) {
    std::cerr << "[ring_allgather_ipc rank " << rank << "] " << msg
              << std::endl;
  }
}

void ensure_ishmem_initialized(int device_index) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.initialized) {
    return;
  }
  int initialized = 0;
  ishmemx_query_initialized(&initialized);
  if (!initialized) {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    ishmemx_attr_t attr;
    attr.device_idx = device_index;
    attr.gpu = true;
    attr.initialize_runtime = !mpi_initialized;
    ishmemx_init_attr(&attr);
  }
  state.initialized = true;
}

// Build (once) the node topology so we can classify each hop's transport.
const Topology& ensure_topology() {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.topo_ready) {
    // validate=false: we only need the classification, not the strict
    // switch/numa containment assertion (a mis-probe should degrade to NIC,
    // not abort the collective).
    state.topo = Topology::from_env(/*validate=*/false);
    state.topo_ready = true;
  }
  return state.topo;
}

void ensure_symmetric(size_t bytes) {
  constexpr size_t kMinBytes = 8 * 1024 * 1024;
  const size_t alloc = std::max(bytes, kMinBytes);
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.symm_bytes >= bytes) {
    return;
  }
  if (state.symm != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.symm);
    ishmem_barrier_all();
    state.symm = nullptr;
    state.symm_bytes = 0;
  }
  state.symm = ishmem_malloc(alloc);
  TORCH_CHECK(
      state.symm != nullptr,
      "ring_allgather_ishmem_ipc: ishmem_malloc failed for ",
      alloc,
      " bytes");
  state.symm_bytes = alloc;
  ishmem_barrier_all();
}

void* current_symmetric() {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.symm;
}

uint64_t* ensure_pad(int slots, sycl::queue& queue) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.pad != nullptr && state.pad_slots >= slots) {
    return state.pad;
  }
  if (state.pad != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.pad);
    ishmem_barrier_all();
    state.pad = nullptr;
    state.pad_slots = 0;
  }
  const size_t bytes = static_cast<size_t>(slots) * sizeof(uint64_t);
  state.pad = static_cast<uint64_t*>(ishmem_malloc(bytes));
  TORCH_CHECK(
      state.pad != nullptr,
      "ring_allgather_ishmem_ipc: ishmem_malloc failed for signal pad (",
      bytes,
      " bytes)");
  state.pad_slots = slots;
  queue.memset(state.pad, 0, bytes).wait_and_throw();
  ishmem_barrier_all();
  return state.pad;
}

constexpr int32_t RING_MAX_WG = 64;

inline void compute_launch(
    int64_t shard_bytes,
    int64_t threads,
    int32_t& num_wg,
    int64_t& slice_bytes) {
  constexpr int64_t kAlign = 16;
  const int64_t per_wg = threads * kAlign;
  int64_t nwg = (shard_bytes + per_wg - 1) / per_wg;
  if (nwg < 1) nwg = 1;
  if (nwg > RING_MAX_WG) nwg = RING_MAX_WG;
  int64_t sb = (shard_bytes + nwg - 1) / nwg;
  sb = ((sb + kAlign - 1) / kAlign) * kAlign;
  nwg = (shard_bytes + sb - 1) / sb;
  if (nwg < 1) nwg = 1;
  num_wg = static_cast<int32_t>(nwg);
  slice_bytes = sb;
}

// 16-byte vectorized work-group copy of `n` bytes (falls back to per-byte for
// the unaligned tail / non-aligned pointers).
inline void wg_copy_bytes(
    const uint8_t* src,
    uint8_t* dst,
    int64_t n,
    int32_t lid,
    int32_t lsize) {
  if (n <= 0) return;
  using vec16 = sycl::vec<uint32_t, 4>;
  const uintptr_t a =
      reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst);
  if ((a % sizeof(vec16)) == 0 && n >= static_cast<int64_t>(sizeof(vec16))) {
    const int64_t nv = n / static_cast<int64_t>(sizeof(vec16));
    auto sv = reinterpret_cast<const vec16*>(src);
    auto dv = reinterpret_cast<vec16*>(dst);
    for (int64_t i = lid; i < nv; i += lsize) dv[i] = sv[i];
    for (int64_t i = nv * static_cast<int64_t>(sizeof(vec16)) + lid; i < n;
         i += lsize)
      dst[i] = src[i];
  } else {
    for (int64_t i = lid; i < n; i += lsize) dst[i] = src[i];
  }
}

// Single-kernel flat-ring allgather with per-hop transport selection. Same
// pipeline as RingAllgatherIshmemSingleKernel, but the push to the right
// neighbour is either a DIRECT peer copy (when `right_data_direct` != null,
// i.e. the right neighbour is PCIe/IPC-reachable) or an ISHMEM RDMA put (NIC).
// The flag is likewise written directly (release fence) on the PCIe path and
// via ishmem_uint64_atomic_set on the NIC path.
struct RingAllgatherIpcSingleKernel {
  uint8_t* symm_base;             // our symmetric data region
  uint64_t* pad;                  // our symmetric signal pad
  uint8_t* right_data_direct;     // direct ptr to right's symm, or null (=NIC)
  uint64_t* right_pad_direct;     // direct ptr to right's pad, or null (=NIC)
  int64_t shard_bytes;
  int64_t slice_bytes;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t num_wg;
  int32_t left_is_nic;            // how the LEFT peer pushes to us (1=NIC)
  uint64_t tag;

  inline void push_slice(
      sycl::nd_item<1> item,
      int64_t off,
      int64_t cnt,
      int32_t slot,
      int32_t lid) const {
    auto grp = item.get_group();
    if (right_data_direct != nullptr) {
      // ---- PCIe / Xe-Link direct peer copy ----
      const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));
      wg_copy_bytes(
          symm_base + off, right_data_direct + off, cnt, lid, lsize);
      // Make the copied bytes visible to the peer device before the flag.
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0) {
        right_pad_direct[slot] = tag;
        sycl::atomic_fence(
            sycl::memory_order::release, sycl::memory_scope::system);
      }
    } else {
      // ---- NIC (ISHMEM RDMA put + separate flag) ----
      ishmemx_putmem_nbi_work_group(
          static_cast<void*>(symm_base + off),
          static_cast<const void*>(symm_base + off),
          static_cast<size_t>(cnt),
          right,
          grp);
      // Ensure the data slice has landed remotely before the flag is written.
      ishmemx_quiet_work_group(grp);
      if (lid == 0) {
        ishmem_uint64_atomic_set(pad + slot, tag, right);
      }
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));

    const int64_t base = static_cast<int64_t>(wg) * slice_bytes;
    int64_t cnt = shard_bytes - base;
    if (cnt > slice_bytes) cnt = slice_bytes;
    if (cnt < 0) cnt = 0;

    // Phase 0: push our own slot `rank`.
    {
      const int64_t off = static_cast<int64_t>(rank) * shard_bytes + base;
      push_slice(item, off, cnt, 0 * num_wg + wg, lid);
    }

    // Steps 1..ws-1: forward the slice the left peer just delivered.
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        // The wait primitive MUST match how the LEFT peer pushed to us,
        // because within a hop the data and the flag travel the SAME path and
        // their ordering only holds under that path's semantics:
        //   - LEFT on NIC: it delivered our slice via RDMA put and bumped this
        //     pad via ishmem_uint64_atomic_set. Only ishmem_uint64_wait_until
        //     correctly observes that NIC-delivered signal and orders it after
        //     the RDMA data.
        //   - LEFT on PCIe/IPC: it directly stored our slice AND then directly
        //     stored this (local) pad, each followed by a system-scope RELEASE
        //     fence. The matching consumer is a system-scope ACQUIRE fence +
        //     plain load spin -- that release/acquire pair is what guarantees
        //     "data visible before flag". ishmem_uint64_wait_until is NOT a
        //     valid pair for a raw PCIe store and cannot guarantee that order.
        uint64_t* slot_ptr = pad + (t - 1) * num_wg + wg;
        if (left_is_nic) {
          ishmem_uint64_wait_until(slot_ptr, ISHMEM_CMP_EQ, tag);
        } else {
          for (;;) {
            sycl::atomic_fence(
                sycl::memory_order::acquire, sycl::memory_scope::system);
            if (*slot_ptr == tag) break;
          }
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
      // Make the left peer's delivered slice (RDMA put or IPC direct store)
      // visible to all work-items (the leader's wait only orders the leader).
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      if (t < world_size - 1) {
        const int32_t idx = (rank - t + world_size) % world_size;
        const int64_t off = static_cast<int64_t>(idx) * shard_bytes + base;
        push_slice(item, off, cnt, t * num_wg + wg, lid);
      }
    }

    // Complete any device-issued ISHMEM puts / the final atomic_set flag before
    // the host-side ishmem_barrier_all() recycles the pads with the next tag.
    // No-op cost on the pure-PCIe (direct) path.
    ishmemx_quiet_work_group(item.get_group());
  }
};

at::Tensor ring_allgather_ishmem_ipc(
    const at::Tensor& input_shard,
    at::Tensor gathered_out,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      input_shard.dim() == 2, "ring_allgather_ishmem_ipc: input_shard must be 2D");
  TORCH_CHECK(
      input_shard.is_contiguous(),
      "ring_allgather_ishmem_ipc: input_shard must be contiguous");
  TORCH_CHECK(
      gathered_out.is_contiguous(),
      "ring_allgather_ishmem_ipc: gathered_out must be contiguous");
  TORCH_CHECK(
      input_shard.scalar_type() == gathered_out.scalar_type(),
      "ring_allgather_ishmem_ipc: dtype mismatch");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "ring_allgather_ishmem_ipc: rank must be in [0, world_size)");
  TORCH_CHECK(
      gathered_out.numel() == input_shard.numel() * world_size,
      "ring_allgather_ishmem_ipc: gathered_out numel must equal input_shard.numel() * world_size");

  if (input_shard.numel() == 0 || world_size <= 1) {
    if (world_size == 1) {
      gathered_out.copy_(input_shard.reshape(gathered_out.sizes()));
    }
    return gathered_out;
  }

  c10::Device device(c10::DeviceType::XPU, input_shard.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  ensure_ishmem_initialized(input_shard.device().index());
  TORCH_CHECK(
      ishmem_my_pe() == rank,
      "ring_allgather_ishmem_ipc: ISHMEM PE does not match rank");
  TORCH_CHECK(
      ishmem_n_pes() == world_size,
      "ring_allgather_ishmem_ipc: ISHMEM PE count does not match world_size");

  const int32_t right = static_cast<int32_t>((rank + 1) % world_size);
  const int32_t left =
      static_cast<int32_t>((rank - 1 + world_size) % world_size);

  // Classify both hops from the topology. Transport is symmetric (it depends
  // only on the symmetric same-switch/same-numa/same-node relation and the same
  // env policy on every rank), and IPC-mappability is symmetric too, so the way
  // WE classify the left hop matches how the LEFT peer actually pushes to us.
  const Topology& topo = ensure_topology();
  Transport right_tp = topo.transport_to(right);
  Transport left_tp = topo.transport_to(left);

  const size_t shard_bytes =
      static_cast<size_t>(input_shard.numel() * input_shard.element_size());
  const size_t gathered_bytes = shard_bytes * static_cast<size_t>(world_size);
  ensure_symmetric(gathered_bytes);
  auto* symm_base = static_cast<uint8_t*>(current_symmetric());

  constexpr int64_t threads = 256;
  int32_t num_wg = 1;
  int64_t slice_bytes = static_cast<int64_t>(shard_bytes);
  compute_launch(
      static_cast<int64_t>(shard_bytes), threads, num_wg, slice_bytes);
  auto* pad = ensure_pad(static_cast<int>(world_size) * RING_MAX_WG, queue);

  // For a PCIe hop, obtain direct (IPC-mapped) pointers to the right peer's
  // symmetric data and pad. ishmem_ptr returns null if the peer is not directly
  // accessible -- in that case we fall back to the NIC (ISHMEM) path.
  uint8_t* right_data_direct = nullptr;
  uint64_t* right_pad_direct = nullptr;
  if (right_tp == Transport::PCIe) {
    right_data_direct =
        static_cast<uint8_t*>(ishmem_ptr(symm_base, right));
    right_pad_direct = static_cast<uint64_t*>(ishmem_ptr(pad, right));
    if (right_data_direct == nullptr || right_pad_direct == nullptr) {
      // Not IPC-mappable after all -> use NIC path.
      right_data_direct = nullptr;
      right_pad_direct = nullptr;
    }
  }
  debug_log(
      rank,
      right_data_direct != nullptr ? "right hop: PCIe direct copy"
                                   : "right hop: NIC ishmem put");

  // Determine how the LEFT peer pushes to us, so the wait primitive can match.
  // Mirror the exact right-hop decision (topology says PCIe AND the peer is
  // IPC-mappable); by symmetry this equals the left peer's own choice.
  int32_t left_is_nic = 1;
  if (left_tp == Transport::PCIe) {
    void* ld = ishmem_ptr(symm_base, left);
    void* lp = ishmem_ptr(pad, left);
    if (ld != nullptr && lp != nullptr) left_is_nic = 0;
  }
  debug_log(rank, left_is_nic ? "left hop: NIC (wait_until)"
                              : "left hop: PCIe (acquire spin)");

  // Seed our own slot.
  const int64_t local_offset = static_cast<int64_t>(rank) * shard_bytes;
  sycl::event dep = queue.memcpy(
      symm_base + local_offset, input_shard.data_ptr(), shard_bytes);

  uint64_t tag;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    tag = ++state.iteration;
  }

  // Every PE must agree the pads are ready before the ring starts pushing
  // (the direct path writes a peer's pad the moment it is signaled).
  ishmem_barrier_all();

  debug_log(rank, "launch single-kernel ring (ipc)");
  auto ring_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dep);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(num_wg) * threads),
            sycl::range<1>(threads)),
        RingAllgatherIpcSingleKernel{
            symm_base,
            pad,
            right_data_direct,
            right_pad_direct,
            static_cast<int64_t>(shard_bytes),
            slice_bytes,
            static_cast<int32_t>(rank),
            static_cast<int32_t>(world_size),
            right,
            num_wg,
            left_is_nic,
            tag});
  });

  auto out_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(ring_event);
    cgh.memcpy(gathered_out.data_ptr(), symm_base, gathered_bytes);
  });
  out_event.wait_and_throw();
  ishmem_barrier_all();
  debug_log(rank, "return");

  return gathered_out;
}

void ring_allgather_ishmem_ipc_finalize(const at::Tensor&) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.symm != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.symm);
    state.symm = nullptr;
    state.symm_bytes = 0;
  }
  if (state.pad != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.pad);
    state.pad = nullptr;
    state.pad_slots = 0;
  }
  if (state.initialized) {
    int initialized = 0;
    ishmemx_query_initialized(&initialized);
    if (initialized) {
      ishmem_finalize();
    }
    state.initialized = false;
  }
}

} // namespace

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ring_allgather_ishmem_ipc(Tensor input_shard, Tensor(a!) gathered_out, "
      "int rank, int world_size) -> Tensor(a!)");
  m.def("ring_allgather_ishmem_ipc_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_allgather_ishmem_ipc", ring_allgather_ishmem_ipc);
  m.impl("ring_allgather_ishmem_ipc_finalize", ring_allgather_ishmem_ipc_finalize);
}
