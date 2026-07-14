// RingAllgatherIshmem.cpp
//
// Standalone ring-allgather implemented purely with ISHMEM device/host APIs,
// as a SINGLE on-device kernel (modeled after RingAllgather.cpp's
// RingAllgatherSingleKernel). The entire ring -- all world_size-1 hops -- runs
// inside ONE kernel launch. Faithfully mirroring the P2P reference, the shard
// is split into `num_wg` contiguous byte-slices and EACH work-group runs a
// fully independent ring pipeline over its own slice: per step the work-group
// cooperatively PUSHes one slice to the right neighbour (data put), a
// work-group quiet guarantees the data has landed remotely, and only then the
// leader writes a SEPARATE flag into the neighbour's signal pad; the leader
// waits on its own signal-pad slot before forwarding a freshly-received slice.
// Because work-groups only ever wait on the SAME wg index of the LEFT peer and
// only signal the SAME wg index of the RIGHT peer, they never wait on each
// other and the kernel cannot deadlock. There are no per-step host launches and
// no per-step host barriers -- cross-rank step ordering is done on-device with
// ISHMEM signal pads, with all communication going through the ISHMEM NIC
// (IBGDA) path.
//
// Ring schedule (bandwidth-optimal, no receiver incast):
//   The symmetric buffer holds `world_size` slots of `shard_bytes` each; slot r
//   ends up holding PE r's shard. Every PE seeds its own slot, then for
//   world_size-1 steps sends one slot to its RIGHT neighbour and receives one
//   from its LEFT neighbour. At step t a PE forwards the block the left peer
//   just delivered:
//       phase 0 : push our own block `rank`  -> right; quiet; flag right pad
//       step  t : wait our pad[t-1] (left delivered block idx=(rank-t)%ws),
//                 then (t < ws-1) forward block idx -> right; quiet; flag pad
//   Signal-pad slot layout (per PE): slot(phase, wg) = phase * num_wg + wg.
//   After world_size-1 steps every slot is present on every PE.
//
// Only ISHMEM APIs are used for communication:
//   - ishmemx_putmem_nbi_work_group  (work-group-collective RDMA write of one
//                                     slice to the neighbour -- data only)
//   - ishmemx_quiet_work_group       (device-side completion of the data put,
//                                     ordering the flag AFTER the data)
//   - ishmem_uint64_atomic_set       (leader writes the separate flag to the
//                                     neighbour's pad)
//   - ishmem_uint64_wait_until       (device-side wait on our own pad)
//   - ishmem_malloc / ishmem_free / ishmem_barrier_all (symmetric heap)
//
// Registered op: symm_mem::ring_allgather_ishmem

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <comm/SYCLHelpers.h>
#include <ishmem.h>
#include <ishmemx.h>
#include <mpi.h>

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
  uint64_t* pad = nullptr;  // symmetric signal pad, world_size uint64 entries
  int pad_pes = 0;
  uint64_t iteration = 0;   // strictly-increasing signal tag (never reused)
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
  return env_enabled("RING_ALLGATHER_ISHMEM_DEBUG");
}

void debug_log(int64_t rank, const char* msg) {
  if (debug_enabled()) {
    std::cerr << "[ring_allgather_ishmem rank " << rank << "] " << msg
              << std::endl;
  }
}

// Lazily bring up ISHMEM. Safe to co-exist with another extension that also
// initialises ISHMEM: we only call init if nobody else has.
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

// Ensure the symmetric buffer can hold `bytes`. Collective (all PEs must call
// with the same size in the same order).
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
      "ring_allgather_ishmem: ishmem_malloc failed for ",
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

// Ensure the symmetric signal pad holds `slots` uint64 entries, zero-init.
// Collective on first allocation / resize. The pad is written across PEs with
// the ISHMEM put-with-signal carried by each transfer and polled on-device
// with ishmem_uint64_wait_until.
uint64_t* ensure_pad(int slots, sycl::queue& queue) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.pad != nullptr && state.pad_pes >= slots) {
    return state.pad;
  }
  if (state.pad != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.pad);
    ishmem_barrier_all();
    state.pad = nullptr;
    state.pad_pes = 0;
  }
  const size_t bytes = static_cast<size_t>(slots) * sizeof(uint64_t);
  state.pad = static_cast<uint64_t*>(ishmem_malloc(bytes));
  TORCH_CHECK(
      state.pad != nullptr,
      "ring_allgather_ishmem: ishmem_malloc failed for signal pad (",
      bytes,
      " bytes)");
  state.pad_pes = slots;
  // Zero the pad once; the strictly-increasing tag means it never needs
  // clearing again between calls.
  queue.memset(state.pad, 0, bytes).wait_and_throw();
  ishmem_barrier_all();
  return state.pad;
}

// Upper bound on work-groups. The signal pad is sized world_size * this so
// the runtime (shard-dependent) num_wg can vary per call without reallocating,
// and must never exceed this value.
constexpr int32_t RING_MAX_WG = 64;

// Deterministic (rank-independent) work-group count / byte-slice for a shard.
// Must be identical on every PE because the signal-pad slot layout
// (slot = phase * num_wg + wg) depends on it; shard_bytes is identical across
// PEs (same input shape), so this is safe.
inline void compute_launch(
    int64_t shard_bytes,
    int64_t threads,
    int32_t& num_wg,
    int64_t& slice_bytes) {
  constexpr int64_t kAlign = 16;  // keep every slice offset 16B-aligned
  const int64_t per_wg = threads * kAlign;
  int64_t nwg = (shard_bytes + per_wg - 1) / per_wg;
  if (nwg < 1) nwg = 1;
  if (nwg > RING_MAX_WG) nwg = RING_MAX_WG;
  int64_t sb = (shard_bytes + nwg - 1) / nwg;
  sb = ((sb + kAlign - 1) / kAlign) * kAlign;  // round up to alignment
  nwg = (shard_bytes + sb - 1) / sb;
  if (nwg < 1) nwg = 1;
  num_wg = static_cast<int32_t>(nwg);
  slice_bytes = sb;
}

// Single-kernel ISHMEM ring allgather. Faithfully mirrors
// RingAllgatherSingleKernel: the shard/slot is split into `num_wg` contiguous
// byte-slices, one per work-group, and EACH work-group runs a fully
// independent ring pipeline over its own slice (it only ever waits on the SAME
// wg index of the LEFT peer and only signals the SAME wg index of the RIGHT
// peer, so work-groups never wait on each other and the kernel cannot
// deadlock). The P2P direct load/store copies are replaced by the ISHMEM
// work-group-collective data put (all work-items of the group cooperate to
// issue one RDMA transfer), and the P2P signal-pad stores are replaced by a
// SEPARATE flag put: a work-group quiet guarantees the data has landed, then
// the leader writes the neighbour's pad with ishmem_uint64_atomic_set.
// Signal-pad layout (per PE): slot(phase, wg) = phase * num_wg + wg.
struct RingAllgatherIshmemSingleKernel {
  uint8_t* symm_base;   // symmetric data region: world_size slots
  uint64_t* pad;        // symmetric signal pad: world_size * num_wg uint64
  int64_t shard_bytes;
  int64_t slice_bytes;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t num_wg;
  uint64_t tag;

  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));

    // This work-group's byte-slice of a single world_size-strided slot.
    const int64_t base = static_cast<int64_t>(wg) * slice_bytes;
    int64_t cnt = shard_bytes - base;
    if (cnt > slice_bytes) cnt = slice_bytes;
    if (cnt < 0) cnt = 0;

    // Phase 0: the whole work-group cooperatively pushes our own slot `rank`
    // slice into the right neighbour's same slot (data put), then a work-group
    // quiet guarantees that data has landed remotely, and only THEN the leader
    // writes the flag: bump the neighbour's pad[0*num_wg+wg] to `tag`. Keeping
    // the data put and the flag put separate makes the "data before flag"
    // ordering explicit via the intervening quiet.
    {
      const int64_t off = static_cast<int64_t>(rank) * shard_bytes + base;
      ishmemx_putmem_nbi_work_group(
          static_cast<void*>(symm_base + off),
          static_cast<const void*>(symm_base + off),
          static_cast<size_t>(cnt),
          right,
          grp);
      // Ensure the data slice has landed on the right neighbour before the flag.
      ishmemx_quiet_work_group(grp);
      if (lid == 0) {
        ishmem_uint64_atomic_set(pad + (0 * num_wg + wg), tag, right);
      }
    }

    // Steps 1..ws-1: at step t the left peer has delivered slot
    // idx=(rank-t) mod ws into our buffer and bumped our pad[(t-1)*num_wg+wg].
    // Wait on that signal, then (unless it is the final slice for us) forward
    // the same slice to the right neighbour and signal its pad[t*num_wg+wg].
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        ishmem_uint64_wait_until(pad + ((t - 1) * num_wg + wg), ISHMEM_CMP_EQ, tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      // Make the left peer's RDMA-delivered slice visible to all work-items
      // (the leader's wait_until only orders the leader).
      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);

      if (t < world_size - 1) {
        const int32_t idx = (rank - t + world_size) % world_size;
        const int64_t off = static_cast<int64_t>(idx) * shard_bytes + base;
        // Data put, then quiet, then the leader writes the flag (see phase 0).
        ishmemx_putmem_nbi_work_group(
            static_cast<void*>(symm_base + off),
            static_cast<const void*>(symm_base + off),
            static_cast<size_t>(cnt),
            right,
            grp);
        ishmemx_quiet_work_group(grp);
        if (lid == 0) {
          ishmem_uint64_atomic_set(pad + (t * num_wg + wg), tag, right);
        }
      }
    }

    // Complete all device-issued puts locally before the host-side
    // ishmem_barrier_all() recycles the pads with the next call's tag.
    ishmemx_quiet_work_group(grp);
  }
};

at::Tensor ring_allgather_ishmem(
    const at::Tensor& input_shard,
    at::Tensor gathered_out,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      input_shard.dim() == 2, "ring_allgather_ishmem: input_shard must be 2D");
  TORCH_CHECK(
      input_shard.is_contiguous(),
      "ring_allgather_ishmem: input_shard must be contiguous");
  TORCH_CHECK(
      gathered_out.is_contiguous(),
      "ring_allgather_ishmem: gathered_out must be contiguous");
  TORCH_CHECK(
      input_shard.scalar_type() == gathered_out.scalar_type(),
      "ring_allgather_ishmem: dtype mismatch");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "ring_allgather_ishmem: rank must be in [0, world_size)");
  TORCH_CHECK(
      gathered_out.numel() == input_shard.numel() * world_size,
      "ring_allgather_ishmem: gathered_out numel must equal input_shard.numel() * world_size");

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
      "ring_allgather_ishmem: ISHMEM PE does not match rank");
  TORCH_CHECK(
      ishmem_n_pes() == world_size,
      "ring_allgather_ishmem: ISHMEM PE count does not match world_size");

  const size_t shard_bytes =
      static_cast<size_t>(input_shard.numel() * input_shard.element_size());
  const size_t gathered_bytes = shard_bytes * static_cast<size_t>(world_size);
  ensure_symmetric(gathered_bytes);
  auto* symm_base = static_cast<uint8_t*>(current_symmetric());

  const int32_t right = static_cast<int32_t>((rank + 1) % world_size);
  constexpr int64_t threads = 256;

  // Deterministic (same on every PE) work-group count / byte-slice, and the
  // matching pad slot count (world_size phases * num_wg).
  int32_t num_wg = 1;
  int64_t slice_bytes = static_cast<int64_t>(shard_bytes);
  compute_launch(
      static_cast<int64_t>(shard_bytes), threads, num_wg, slice_bytes);
  auto* pad = ensure_pad(
      static_cast<int>(world_size) * RING_MAX_WG, queue);

  // Seed our own slot.
  const int64_t local_offset = static_cast<int64_t>(rank) * shard_bytes;
  sycl::event dep = queue.memcpy(
      symm_base + local_offset, input_shard.data_ptr(), shard_bytes);

  // Fresh strictly-increasing signal tag for this call (pads never reused).
  uint64_t tag;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    tag = ++state.iteration;
  }
  debug_log(rank, "launch single-kernel ring");
  // Entire ring runs in ONE kernel launch; each work-group drives an
  // independent ring pipeline over its slice with work-group-collective ISHMEM
  // put-with-signal and an on-device wait on its own pad slot.
  auto ring_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dep);
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(num_wg) * threads),
            sycl::range<1>(threads)),
        RingAllgatherIshmemSingleKernel{
            symm_base,
            pad,
            static_cast<int64_t>(shard_bytes),
            slice_bytes,
            static_cast<int32_t>(rank),
            static_cast<int32_t>(world_size),
            right,
            num_wg,
            tag});
  });

  // The ring kernel issues all device-side ISHMEM put-with-signal transfers.
  // The host-side ishmem_barrier_all() below is the cross-call fence that lets
  // the reused signal pads (world_size * num_wg slots) be recycled with the
  // next strictly-increasing tag. That fence is only meaningful once the kernel
  // that produces / consumes this call's pad values has actually completed on
  // the device -- otherwise a pipelined next call (e.g. a timed loop with no
  // per-iteration synchronize) can signal a pad slot with the next tag before a
  // peer's wait_until(EQ, this_tag) observes it, overwriting the value it is
  // spinning on and deadlocking. So we chain the copy-out on the ring kernel
  // and wait for it below.
  debug_log(rank, "ring kernel done");

  auto out_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(ring_event);
    cgh.memcpy(gathered_out.data_ptr(), symm_base, gathered_bytes);
  });
  out_event.wait_and_throw();
  // Cross-call safety: make sure every PE has finished consuming this call's
  // signal pads / slots before any PE reuses them with the next tag.
  ishmem_barrier_all();
  debug_log(rank, "return");

  return gathered_out;
}

void ring_allgather_ishmem_finalize(const at::Tensor&) {
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
    state.pad_pes = 0;
  }
  if (state.initialized) {
    // Only finalize if we are the ones who still consider ISHMEM live. If
    // another extension owns the runtime it will finalize on its own; a double
    // finalize is avoided by the query below.
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
      "ring_allgather_ishmem(Tensor input_shard, Tensor(a!) gathered_out, "
      "int rank, int world_size) -> Tensor(a!)");
  m.def("ring_allgather_ishmem_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_allgather_ishmem", ring_allgather_ishmem);
  m.impl("ring_allgather_ishmem_finalize", ring_allgather_ishmem_finalize);
}
