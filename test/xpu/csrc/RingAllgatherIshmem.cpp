// RingAllgatherIshmem.cpp
//
// Standalone ring-allgather implemented purely with ISHMEM device/host APIs,
// as a SINGLE on-device kernel (modeled after RingAllgather.cpp's
// RingAllgatherSingleKernel). The entire ring -- all world_size-1 hops -- runs
// inside ONE kernel launch: the leader work-item PUSHes one slot to the right
// neighbour per step, quiets to guarantee the RDMA write has landed remotely,
// then sets the neighbour's signal pad; it waits on its own signal pad before
// forwarding a freshly-received slot. There are no per-step host launches and
// no per-step host barriers -- cross-rank step ordering is done on-device with
// ISHMEM signal pads, exactly mirroring the P2P reference but with all
// communication going through the ISHMEM NIC (IBGDA) path.
//
// Ring schedule (bandwidth-optimal, no receiver incast):
//   The symmetric buffer holds `world_size` slots of `shard_bytes` each; slot r
//   ends up holding PE r's shard. Every PE seeds its own slot, then for
//   world_size-1 steps sends one slot to its RIGHT neighbour and receives one
//   from its LEFT neighbour. At step t a PE forwards the block the left peer
//   just delivered:
//       phase 0 : push our own block `rank`  -> right; signal right pad[0]
//       step  t : wait our pad[t-1] (left delivered block idx=(rank-t)%ws),
//                 then (t < ws-1) forward block idx -> right; signal pad[t]
//   After world_size-1 steps every slot is present on every PE.
//
// Only ISHMEM APIs are used for communication:
//   - ishmem_putmem_nbi        (GPU-issued RDMA write of one slot to neighbour)
//   - ishmem_quiet             (device-side completion of outstanding puts)
//   - ishmem_uint64_atomic_set (device-side signal to neighbour's pad)
//   - ishmem_uint64_wait_until (device-side wait on our own pad)
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

// Ensure the symmetric signal pad holds `pes` uint64 entries, zero-initialised.
// Collective on first allocation / resize. The pad is written across PEs with
// device-side ishmem atomic_set and polled with ishmem_uint64_wait_until.
uint64_t* ensure_pad(int pes, sycl::queue& queue) {
  auto& state = get_state();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.pad != nullptr && state.pad_pes >= pes) {
    return state.pad;
  }
  if (state.pad != nullptr) {
    ishmem_barrier_all();
    ishmem_free(state.pad);
    ishmem_barrier_all();
    state.pad = nullptr;
    state.pad_pes = 0;
  }
  const size_t bytes = static_cast<size_t>(pes) * sizeof(uint64_t);
  state.pad = static_cast<uint64_t*>(ishmem_malloc(bytes));
  TORCH_CHECK(
      state.pad != nullptr,
      "ring_allgather_ishmem: ishmem_malloc failed for signal pad (",
      bytes,
      " bytes)");
  state.pad_pes = pes;
  // Zero the pad once; the strictly-increasing tag means it never needs
  // clearing again between calls.
  queue.memset(state.pad, 0, bytes).wait_and_throw();
  ishmem_barrier_all();
  return state.pad;
}

// Single-kernel ISHMEM ring allgather. Only the leader work-item runs the ring;
// all communication is via ISHMEM device APIs (NIC/IBGDA path). Mirrors
// RingAllgatherSingleKernel's phase-0-then-forward pipeline, with ISHMEM
// signal pads replacing the P2P signal-pad stores.
struct RingAllgatherIshmemSingleKernel {
  uint8_t* symm_base;   // symmetric data region: world_size slots
  uint64_t* pad;        // symmetric signal pad: world_size uint64 (same VA all PEs)
  int64_t shard_bytes;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  uint64_t tag;

  void operator()(sycl::nd_item<1> item) const {
    // Leader work-item of the leader work-group drives the entire ring.
    if (item.get_global_id(0) != 0) {
      return;
    }

    // Phase 0: push our own block `rank` to the right neighbour, quiet so the
    // RDMA write has landed remotely, then signal the neighbour's pad[0].
    {
      const int64_t off = static_cast<int64_t>(rank) * shard_bytes;
      ishmem_putmem_nbi(
          static_cast<void*>(symm_base + off),
          static_cast<const void*>(symm_base + off),
          static_cast<size_t>(shard_bytes),
          right);
      ishmem_quiet();
      ishmem_uint64_atomic_set(pad + 0, tag, right);
    }

    // Steps 1..ws-1: wait for the left peer to deliver block idx=(rank-t)%ws
    // into our slot, then forward it to the right (the last received block is
    // final for us and is not forwarded).
    for (int32_t t = 1; t < world_size; ++t) {
      ishmem_uint64_wait_until(pad + (t - 1), ISHMEM_CMP_EQ, tag);
      if (t < world_size - 1) {
        const int32_t idx = (rank - t + world_size) % world_size;
        const int64_t off = static_cast<int64_t>(idx) * shard_bytes;
        ishmem_putmem_nbi(
            static_cast<void*>(symm_base + off),
            static_cast<const void*>(symm_base + off),
            static_cast<size_t>(shard_bytes),
            right);
        ishmem_quiet();
        ishmem_uint64_atomic_set(pad + t, tag, right);
      }
    }
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
  auto* pad = ensure_pad(static_cast<int>(world_size), queue);

  // Seed our own slot.
  const int64_t local_offset = static_cast<int64_t>(rank) * shard_bytes;
  sycl::event dep = queue.memcpy(
      symm_base + local_offset, input_shard.data_ptr(), shard_bytes);

  const int32_t right = static_cast<int32_t>((rank + 1) % world_size);
  constexpr int64_t threads = 256;

  // Fresh strictly-increasing signal tag for this call (pads never reused).
  uint64_t tag;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    tag = ++state.iteration;
  }

  debug_log(rank, "launch single-kernel ring");
  print("start to do ring allgather", flush=True);
  // Entire ring runs in ONE kernel launch; the leader work-item drives all
  // world_size-1 hops with device-side ISHMEM put/quiet/signal/wait.
  auto ring_event = queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dep);
    cgh.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(threads), sycl::range<1>(threads)),
        RingAllgatherIshmemSingleKernel{
            symm_base,
            pad,
            static_cast<int64_t>(shard_bytes),
            static_cast<int32_t>(rank),
            static_cast<int32_t>(world_size),
            right,
            tag});
  });
  //ring_event.wait_and_throw();
  debug_log(rank, "ring kernel done");

  auto out_event = queue.submit([&](sycl::handler& cgh) {
    cgh.memcpy(gathered_out.data_ptr(), symm_base, gathered_bytes);
  });
  //out_event.wait_and_throw();
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
