#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

// ===========================================================================
// Ring reduce-scatter using symmetric memory  --  SINGLE-KERNEL, PUSH based.
//
// This mirrors the pipelined ring algorithm used by oneCCL's
// reduce_scatter_large_sycl_ring: every rank starts with the full input
// ([world_size, chunk]); over the ring steps the partial sum for each block is
// accumulated one hop at a time, so each link only carries one chunk per step
// (instead of an all-to-all fan-in).  Data moves by PUSH: each rank WRITES the
// running partial into its right neighbor's `acc` buffer (remote stores are
// far cheaper than remote loads on Xe-Link / PCIe).
//
// Topology (push based): right = (rank + 1) % ws.  The partial sum for final
// block b flows b+1 -> b+2 -> ... -> b, gaining one rank's contribution per
// hop, and is finalized on rank b (the standard reduce-scatter mapping where
// rank r receives reduced block r).  See the kernel comment for the per-step
// block schedule.
//
// Cross-rank ordering uses per-step signal pads (system-scope release/acquire)
// just like the oneCCL device-side ring kernels. `iteration` is a strictly
// increasing tag so the signal pads never need to be reset between calls.
// ===========================================================================

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

namespace {

// Upper bound on work-groups; the signal pad is sized world_size * this by
// the Python wrapper, so the actual (chunk-dependent) num_wg must not exceed
// this value.
constexpr int32_t RING_MAX_WG = 64;

// Matches src/xccl/Signal.hpp store_release: write first, THEN issue the
// system-scope release fence so the store is actually flushed to the shared
// coherence point and becomes visible to the peer device.
inline void store_release_sys(uint32_t* addr, uint32_t val) {
  *addr = val;
  sycl::atomic_fence(
      sycl::memory_order::release, sycl::memory_scope::system);
}

// Matches src/xccl/Signal.hpp load_acquire: issue a system-scope acquire fence
// BEFORE EACH load so every iteration re-reads the value from the shared
// coherence point (a plain volatile load is not coherent across devices on
// PCIe and can spin forever on a cached value).
inline void wait_eq_sys(uint32_t* addr, uint32_t val) {
  for (;;) {
    sycl::atomic_fence(
        sycl::memory_order::acquire, sycl::memory_scope::system);
    if (*addr == val)
      break;
  }
}

// SINGLE-KERNEL chunked ring reduce-scatter (PUSH based).
//
// Each work-group owns a contiguous slice [base, base+cnt) of the per-block
// `chunk` dimension and runs an independent ring pipeline over that slice,
// waiting only on the SAME work-group index of the LEFT peer and signaling
// only the SAME work-group index of the RIGHT peer (see RingAllgather.cpp for
// why this avoids deadlocks).  Like the oneCCL reference, data moves by PUSH:
// the running partial sum is WRITTEN into the right peer's `acc` buffer
// (remote stores are far cheaper than remote loads on Xe-Link / PCIe).
//
// The partial sum for final block b travels the ring b+1 -> b+2 -> ... -> b,
// gaining one rank's contribution at each hop.  From rank r's view, at step t
// it handles block b_t = (rank - 1 - t + ws) % ws:
//   - Phase 0 (t=0): push our own input[b0] (b0=(rank-1+ws)%ws) into the
//     right peer's acc[b0]; signal right (slot 0).  No incoming partial yet.
//   - Step t (1..ws-2): wait our OWN signal for phase t-1 (left pushed the
//     partial for block b_t into our acc[b_t]); push input[b_t] + acc[b_t]
//     into the right peer's acc[b_t]; signal right (slot t).
//   - Step ws-1: b_t == rank is final for us; wait phase ws-2, then write
//     input[rank] + acc[rank] into `output` (no push).
//
// `acc` only ever receives remote partials (never pre-initialized); local
// contributions are folded in on the fly.
//
// Signal-pad layout (per rank): slot(phase, wg) = phase * num_wg + wg, with
// phase in [0, ws-1].
template <typename scalar_t, int VEC_SIZE>
struct RingReduceScatterSingleKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_ptr;
  const int64_t* rank_buffers_ptr;
  const int64_t* signal_pads_ptr;
  scalar_t* acc_ptr;
  scalar_t* output_ptr;
  int64_t chunk;
  int64_t elems_per_wg;
  int32_t rank;
  int32_t world_size;
  int32_t left;
  int32_t right;
  int32_t num_wg;
  uint32_t tag;

  inline void wg_copy(
      const scalar_t* src,
      scalar_t* dst,
      int64_t n,
      int32_t lid,
      int32_t lsize) const {
    if (n <= 0) return;
    const uintptr_t a =
        reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst);
    if ((a % (VEC_SIZE * sizeof(scalar_t))) == 0 && n >= VEC_SIZE) {
      const int64_t nv = n / VEC_SIZE;
      auto sv = reinterpret_cast<const vec_t*>(src);
      auto dv = reinterpret_cast<vec_t*>(dst);
      for (int64_t i = lid; i < nv; i += lsize)
        dv[i] = sv[i];
      for (int64_t i = nv * VEC_SIZE + lid; i < n; i += lsize)
        dst[i] = src[i];
    } else {
      for (int64_t i = lid; i < n; i += lsize)
        dst[i] = src[i];
    }
  }

  // dst = a + b (accumulate in float) for `n` elements.
  // Vectorized path: load `a` and `b` as VEC_SIZE-wide vectors, fold lane by
  // lane in float, and issue a single VEC_SIZE-wide store — so the remote
  // `dst` store is one coalesced block transaction (the bandwidth-critical
  // part), exactly like wg_copy.  Falls back to scalar when not VEC-aligned.
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
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    const int64_t base = static_cast<int64_t>(wg) * elems_per_wg;
    int64_t cnt = chunk - base;
    if (cnt > elems_per_wg) cnt = elems_per_wg;
    if (cnt < 0) cnt = 0;

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[rank]);
    uint32_t* right_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[right]);
    scalar_t* right_acc =
        reinterpret_cast<scalar_t*>(rank_buffers_ptr[right]);

    // Phase 0: start the partial for block b0 by pushing our own contribution.
    {
      const int32_t b0 = (rank - 1 + world_size) % world_size;
      const int64_t off = static_cast<int64_t>(b0) * chunk + base;
      wg_copy(input_ptr + off, right_acc + off, cnt, lid, lsize);
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0) {
        store_release_sys(right_pad + (0 * num_wg + wg), tag);
      }
    }

    // Steps 1..ws-1: fold our contribution into the incoming partial and push.
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      const int32_t b_t = (rank - 1 - t + 2 * world_size) % world_size;
      const int64_t off = static_cast<int64_t>(b_t) * chunk + base;

      if (t < world_size - 1) {
        // Forward: right.acc[b_t] = input[b_t] + acc[b_t].
        wg_add2(
            input_ptr + off, acc_ptr + off, right_acc + off, cnt, lid, lsize);
        sycl::atomic_fence(
            sycl::memory_order::release, sycl::memory_scope::system);
        item.barrier(sycl::access::fence_space::local_space);
        if (lid == 0) {
          store_release_sys(right_pad + (t * num_wg + wg), tag);
        }
      } else {
        // Final: block `rank` is fully reduced here (b_t == rank).
        wg_add2(
            input_ptr + off, acc_ptr + off, output_ptr + base, cnt, lid, lsize);
      }
    }
  }
};

// Deterministic (rank-independent) work-group count for a given chunk size.
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

}  // namespace

// `acc` is rank's symmetric-memory buffer and must equal
// rank_buffers_ptr[rank]; peers read it directly.
at::Tensor ring_reduce_scatter(
    const at::Tensor& input,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& signal_pads_ptr,
    at::Tensor acc,
    at::Tensor output,
    int64_t rank,
    int64_t world_size,
    int64_t iteration) {
  TORCH_CHECK(input.dim() == 1, "ring_reduce_scatter: input must be 1D");
  TORCH_CHECK(input.is_contiguous(), "ring_reduce_scatter: input must be contiguous");
  TORCH_CHECK(acc.dim() == 1, "ring_reduce_scatter: acc must be 1D");
  TORCH_CHECK(acc.is_contiguous(), "ring_reduce_scatter: acc must be contiguous");
  TORCH_CHECK(output.dim() == 1, "ring_reduce_scatter: output must be 1D");
  TORCH_CHECK(output.is_contiguous(), "ring_reduce_scatter: output must be contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "ring_reduce_scatter: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "ring_reduce_scatter: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      signal_pads_ptr.dim() == 1 && signal_pads_ptr.size(0) == world_size,
      "ring_reduce_scatter: signal_pads_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      signal_pads_ptr.scalar_type() == at::kLong,
      "ring_reduce_scatter: signal_pads_ptr must be int64");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "ring_reduce_scatter: rank must be in [0, world_size)");
  TORCH_CHECK(
      input.scalar_type() == output.scalar_type() &&
          input.scalar_type() == acc.scalar_type(),
      "ring_reduce_scatter: input/acc/output must have same dtype");
  TORCH_CHECK(iteration > 0, "ring_reduce_scatter: iteration must be > 0");

  const int64_t chunk = output.numel();
  TORCH_CHECK(
      input.numel() == chunk * world_size,
      "ring_reduce_scatter: input.numel() must equal output.numel() * world_size");
  TORCH_CHECK(
      acc.numel() == input.numel(),
      "ring_reduce_scatter: acc.numel() must equal input.numel()");

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

  const int32_t ws = static_cast<int32_t>(world_size);
  const int32_t r = static_cast<int32_t>(rank);
  const int32_t left = (r - 1 + ws) % ws;
  const int32_t right = (r + 1) % ws;
  const uint32_t tag = static_cast<uint32_t>(iteration);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;
  int32_t num_wg = 1;
  int64_t elems_per_wg = chunk;
  compute_launch(chunk, threads, VEC_SIZE, num_wg, elems_per_wg);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(output.scalar_type(), "ring_reduce_scatter", [&]() {
    auto kfn = RingReduceScatterSingleKernel<scalar_t, VEC_SIZE>{
        input.data_ptr<scalar_t>(),
        rank_buffers_ptr.data_ptr<int64_t>(),
        signal_pads_ptr.data_ptr<int64_t>(),
        acc.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        chunk,
        elems_per_wg,
        r,
        ws,
        left,
        right,
        num_wg,
        tag};
    sycl_kernel_submit(
        sycl::range<1>(static_cast<size_t>(num_wg) * threads),
        sycl::range<1>(threads),
        queue,
        kfn);
  });

  return output;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ring_reduce_scatter(Tensor input, Tensor rank_buffers_ptr, "
      "Tensor signal_pads_ptr, Tensor(a!) acc, Tensor(a!) output, int rank, "
      "int world_size, int iteration) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_reduce_scatter", ring_reduce_scatter);
}
