// RingReduceScatterIshmem.cpp
//
// Standalone ring reduce-scatter implemented purely with ISHMEM device/host
// APIs, as a SINGLE on-device kernel. It is a straight port of the P2P
// symmetric-memory kernel in RingReduceScatter.cpp: the pipelined ring
// algorithm and the per-step block schedule are identical, but every
// cross-rank data movement / signalling primitive is replaced by its ISHMEM
// equivalent (modeled after RingAllgatherIshmem.cpp).
//
// Algorithm (same as RingReduceScatter.cpp, PUSH based):
//   Each rank starts with the full input ([world_size, chunk]); over the ring
//   steps the partial sum for each block is accumulated one hop at a time, so
//   each link only carries one chunk per step. Data moves by PUSH: each rank
//   contributes its own block into the running partial and forwards it to its
//   RIGHT neighbour. The partial sum for final block b flows
//   b+1 -> b+2 -> ... -> b, gaining one rank's contribution per hop, and is
//   finalized on rank b (rank r receives reduced block r).
//
// From rank r's view, at step t it handles block b_t = (rank - 1 - t + ws)%ws:
//   - Phase 0 (t=0): PUSH our own input[b0] (b0=(rank-1+ws)%ws) into the right
//     peer's acc[b0]; signal right (slot 0). No incoming partial yet.
//   - Step t (1..ws-2): wait our OWN signal for phase t-1 (left pushed the
//     partial for block b_t into our acc[b_t]); locally compute
//     input[b_t] + acc[b_t] (in place in acc[b_t]), PUSH it into the right
//     peer's acc[b_t]; signal right (slot t).
//   - Step ws-1: b_t == rank is final for us; wait phase ws-2, then write
//     input[rank] + acc[rank] into `output` (no push).
//
// Difference from the P2P kernel: the P2P kernel computes the partial sum
// DIRECTLY into the right peer's `acc` with a remote store. ISHMEM cannot fuse
// a compute with a remote store, so we instead compute the partial sum locally
// (in place, overwriting our own acc[b_t] which we no longer need), then PUSH
// that slice to the right peer with a work-group-collective ISHMEM put. A
// work-group quiet guarantees the data has landed remotely, and only then the
// leader writes a SEPARATE flag into the neighbour's signal pad.
//
// Like the P2P kernel and RingAllgatherIshmem.cpp, the per-block `chunk`
// dimension is split into `num_wg` contiguous element-slices and EACH
// work-group runs a fully independent ring pipeline over its own slice: it
// only ever waits on the SAME wg index of the LEFT peer and only signals the
// SAME wg index of the RIGHT peer, so work-groups never wait on each other and
// the kernel cannot deadlock.
//   Signal-pad slot layout (per PE): slot(phase, wg) = phase * num_wg + wg,
//   with phase in [0, ws-1].
//
// Only ISHMEM APIs are used for communication:
//   - ishmemx_putmem_nbi_work_group  (work-group-collective RDMA write of one
//                                     slice to the neighbour -- data only)
//   - ishmemx_quiet_work_group       (device-side completion of the data put)
//   - ishmem_uint64_atomic_set       (leader writes the separate flag)
//   - ishmem_uint64_wait_until       (device-side wait on our own pad)
//   - ishmem_malloc / ishmem_free / ishmem_barrier_all (symmetric heap)
//
// Registered op: symm_mem::ring_reduce_scatter_ishmem

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
#include <type_traits>

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

namespace {

struct RingState {
  std::mutex mutex;
  bool initialized = false;
  void* symm = nullptr;      // symmetric acc region: world_size * chunk scalars
  size_t symm_bytes = 0;
  uint64_t* pad = nullptr;   // symmetric signal pad, world_size*RING_MAX_WG u64
  int pad_slots = 0;
  uint64_t iteration = 0;    // strictly-increasing signal tag (never reused)
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
  return env_enabled("RING_REDUCE_SCATTER_ISHMEM_DEBUG");
}

void debug_log(int64_t rank, const char* msg) {
  if (debug_enabled()) {
    std::cerr << "[ring_reduce_scatter_ishmem rank " << rank << "] " << msg
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

// Ensure the symmetric acc buffer can hold `bytes`. Collective (all PEs must
// call with the same size in the same order).
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
      "ring_reduce_scatter_ishmem: ishmem_malloc failed for ",
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
// Collective on first allocation / resize.
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
      "ring_reduce_scatter_ishmem: ishmem_malloc failed for signal pad (",
      bytes,
      " bytes)");
  state.pad_slots = slots;
  // Zero the pad once; the strictly-increasing tag means it never needs
  // clearing again between calls.
  queue.memset(state.pad, 0, bytes).wait_and_throw();
  ishmem_barrier_all();
  return state.pad;
}

// Upper bound on work-groups. The signal pad is sized world_size * this so
// the runtime (chunk-dependent) num_wg can vary per call without reallocating,
// and must never exceed this value.
constexpr int32_t RING_MAX_WG = 64;

// Deterministic (rank-independent) work-group count / element-slice for a
// given chunk. Must be identical on every PE because the signal-pad slot
// layout (slot = phase * num_wg + wg) depends on it; chunk is identical across
// PEs (same input shape), so this is safe.
inline void compute_launch(
    int64_t chunk,
    int64_t threads,
    int VEC_SIZE,
    int32_t& num_wg,
    int64_t& elems_per_wg) {
  const int64_t per_wg = threads * VEC_SIZE;
  int64_t nwg = (chunk + per_wg - 1) / per_wg;
  if (nwg < 1) nwg = 1;
  if (nwg > RING_MAX_WG) nwg = RING_MAX_WG;
  int64_t epw = (chunk + nwg - 1) / nwg;
  epw = ((epw + VEC_SIZE - 1) / VEC_SIZE) * VEC_SIZE;  // round up to VEC
  nwg = (chunk + epw - 1) / epw;
  if (nwg < 1) nwg = 1;
  num_wg = static_cast<int32_t>(nwg);
  elems_per_wg = epw;
}

// Single-kernel ISHMEM ring reduce-scatter (PUSH based). Faithful port of
// RingReduceScatterSingleKernel: same block schedule, same per-work-group
// independent ring pipeline. The direct remote store of the fused
// (input + acc) sum is replaced by (1) a LOCAL in-place accumulation into our
// own acc[b_t] slice followed by (2) a work-group-collective ISHMEM put of
// that slice into the right peer's acc[b_t], a quiet, then a SEPARATE flag
// write into the neighbour's pad.
// Signal-pad layout (per PE): slot(phase, wg) = phase * num_wg + wg.
template <typename scalar_t, int VEC_SIZE>
struct RingReduceScatterIshmemSingleKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_ptr;   // local input: world_size * chunk
  scalar_t* acc_ptr;           // our symmetric acc: world_size * chunk
  scalar_t* symm_base;         // == acc_ptr, used as the ISHMEM symmetric base
  scalar_t* output_ptr;        // local output: chunk
  uint64_t* pad;               // symmetric signal pad
  int64_t chunk;
  int64_t elems_per_wg;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t num_wg;
  uint64_t tag;

  // dst[0..n) = a + b, accumulated in float. Vectorized when VEC-aligned.
  // Pure local compute (no remote store), so a plain vectorized store is used.
  inline void wg_add2(
      const scalar_t* a,
      const scalar_t* b,
      scalar_t* dst,
      int64_t n,
      int32_t lid,
      int32_t lsize) const {
    if (n <= 0) return;
    const uintptr_t al = reinterpret_cast<uintptr_t>(a) |
        reinterpret_cast<uintptr_t>(b) | reinterpret_cast<uintptr_t>(dst);
    if ((al % (VEC_SIZE * sizeof(scalar_t))) == 0 && n >= VEC_SIZE) {
      const int64_t nv = n / VEC_SIZE;
      auto av = reinterpret_cast<const vec_t*>(a);
      auto bv = reinterpret_cast<const vec_t*>(b);
      auto dv = reinterpret_cast<vec_t*>(dst);
      for (int64_t i = lid; i < nv; i += lsize) {
        vec_t va = av[i];
        vec_t vb = bv[i];
        vec_t vd;
#pragma unroll
        for (int k = 0; k < VEC_SIZE; ++k) {
          const vec_elem_t ra = va[k];
          const vec_elem_t rb = vb[k];
          const scalar_t sa = *reinterpret_cast<const scalar_t*>(&ra);
          const scalar_t sb = *reinterpret_cast<const scalar_t*>(&rb);
          const scalar_t sres = static_cast<scalar_t>(
              static_cast<float>(sa) + static_cast<float>(sb));
          vd[k] = *reinterpret_cast<const vec_elem_t*>(&sres);
        }
        dv[i] = vd;
      }
      for (int64_t i = nv * VEC_SIZE + lid; i < n; i += lsize) {
        dst[i] = static_cast<scalar_t>(
            static_cast<float>(a[i]) + static_cast<float>(b[i]));
      }
    } else {
      for (int64_t i = lid; i < n; i += lsize) {
        dst[i] = static_cast<scalar_t>(
            static_cast<float>(a[i]) + static_cast<float>(b[i]));
      }
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    auto grp = item.get_group();
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    const int64_t base = static_cast<int64_t>(wg) * elems_per_wg;
    int64_t cnt = chunk - base;
    if (cnt > elems_per_wg) cnt = elems_per_wg;
    if (cnt < 0) cnt = 0;

    // Phase 0: PUSH our own contribution for block b0 into the right peer's
    // acc[b0]. The source (input) is regular device memory; the destination is
    // the symmetric acc region on the right PE. Then quiet + flag.
    {
      const int32_t b0 = (rank - 1 + world_size) % world_size;
      const int64_t off = static_cast<int64_t>(b0) * chunk + base;
      ishmemx_putmem_nbi_work_group(
          static_cast<void*>(symm_base + off),
          static_cast<const void*>(input_ptr + off),
          static_cast<size_t>(cnt) * sizeof(scalar_t),
          right,
          grp);
      ishmemx_quiet_work_group(grp);
      if (lid == 0) {
        ishmem_uint64_atomic_set(pad + (0 * num_wg + wg), tag, right);
      }
    }

    // Steps 1..ws-1: fold our contribution into the incoming partial and push.
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        ishmem_uint64_wait_until(
            pad + ((t - 1) * num_wg + wg), ISHMEM_CMP_EQ, tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      // Make the left peer's RDMA-delivered partial visible to all work-items.
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      const int32_t b_t = (rank - 1 - t + 2 * world_size) % world_size;
      const int64_t off = static_cast<int64_t>(b_t) * chunk + base;

      if (t < world_size - 1) {
        // Local in-place accumulate: acc[b_t] = input[b_t] + acc[b_t].
        wg_add2(input_ptr + off, acc_ptr + off, acc_ptr + off, cnt, lid, lsize);
        // Ensure all lanes finished writing acc[b_t] before the collective put
        // reads it.
        item.barrier(sycl::access::fence_space::local_space);
        // PUSH the freshly accumulated slice into the right peer's acc[b_t].
        ishmemx_putmem_nbi_work_group(
            static_cast<void*>(symm_base + off),
            static_cast<const void*>(symm_base + off),
            static_cast<size_t>(cnt) * sizeof(scalar_t),
            right,
            grp);
        ishmemx_quiet_work_group(grp);
        if (lid == 0) {
          ishmem_uint64_atomic_set(pad + (t * num_wg + wg), tag, right);
        }
      } else {
        // Final: block `rank` is fully reduced here (b_t == rank). No push.
        wg_add2(
            input_ptr + off, acc_ptr + off, output_ptr + base, cnt, lid, lsize);
      }
    }

    // Complete all device-issued puts locally before the host-side
    // ishmem_barrier_all() recycles the pads with the next call's tag.
    ishmemx_quiet_work_group(grp);
  }
};

}  // namespace

at::Tensor ring_reduce_scatter_ishmem(
    const at::Tensor& input,
    at::Tensor output,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(input.dim() == 1, "ring_reduce_scatter_ishmem: input must be 1D");
  TORCH_CHECK(
      input.is_contiguous(),
      "ring_reduce_scatter_ishmem: input must be contiguous");
  TORCH_CHECK(
      output.dim() == 1, "ring_reduce_scatter_ishmem: output must be 1D");
  TORCH_CHECK(
      output.is_contiguous(),
      "ring_reduce_scatter_ishmem: output must be contiguous");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "ring_reduce_scatter_ishmem: rank must be in [0, world_size)");
  TORCH_CHECK(
      input.scalar_type() == output.scalar_type(),
      "ring_reduce_scatter_ishmem: input/output must have same dtype");

  const int64_t chunk = output.numel();
  TORCH_CHECK(
      input.numel() == chunk * world_size,
      "ring_reduce_scatter_ishmem: input.numel() must equal output.numel() * world_size");

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  if (chunk == 0) {
    return output;
  }
  if (world_size == 1) {
    output.copy_(input.narrow(0, 0, chunk));
    return output;
  }

  ensure_ishmem_initialized(input.device().index());
  TORCH_CHECK(
      ishmem_my_pe() == rank,
      "ring_reduce_scatter_ishmem: ISHMEM PE does not match rank");
  TORCH_CHECK(
      ishmem_n_pes() == world_size,
      "ring_reduce_scatter_ishmem: ISHMEM PE count does not match world_size");

  const size_t acc_bytes =
      static_cast<size_t>(input.numel()) * input.element_size();
  ensure_symmetric(acc_bytes);

  const int32_t ws = static_cast<int32_t>(world_size);
  const int32_t r = static_cast<int32_t>(rank);
  const int32_t right = (r + 1) % ws;

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;
  int32_t num_wg = 1;
  int64_t elems_per_wg = chunk;
  compute_launch(chunk, threads, VEC_SIZE, num_wg, elems_per_wg);

  auto* pad = ensure_pad(static_cast<int>(world_size) * RING_MAX_WG, queue);

  // Fresh strictly-increasing signal tag for this call (pads never reused).
  uint64_t tag;
  {
    auto& state = get_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    tag = ++state.iteration;
  }

  debug_log(rank, "launch single-kernel ring reduce-scatter");
  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ring_reduce_scatter_ishmem", [&]() {
        auto* symm_base = static_cast<scalar_t*>(current_symmetric());
        auto ring_event = queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(
                  sycl::range<1>(static_cast<size_t>(num_wg) * threads),
                  sycl::range<1>(threads)),
              RingReduceScatterIshmemSingleKernel<scalar_t, VEC_SIZE>{
                  input.data_ptr<scalar_t>(),
                  symm_base,
                  symm_base,
                  output.data_ptr<scalar_t>(),
                  pad,
                  chunk,
                  elems_per_wg,
                  r,
                  ws,
                  right,
                  num_wg,
                  tag});
        });
        ring_event.wait_and_throw();
      });

  debug_log(rank, "ring kernel done");
  // Cross-call safety: make sure every PE has finished consuming this call's
  // signal pads / slots before any PE reuses them with the next tag.
  ishmem_barrier_all();
  debug_log(rank, "return");

  return output;
}

void ring_reduce_scatter_ishmem_finalize(const at::Tensor&) {
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

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ring_reduce_scatter_ishmem(Tensor input, Tensor(a!) output, "
      "int rank, int world_size) -> Tensor(a!)");
  m.def("ring_reduce_scatter_ishmem_finalize(Tensor dummy) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_reduce_scatter_ishmem", ring_reduce_scatter_ishmem);
  m.impl(
      "ring_reduce_scatter_ishmem_finalize",
      ring_reduce_scatter_ishmem_finalize);
}
