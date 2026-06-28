#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

// ===========================================================================
// Ring allgather using symmetric memory  --  SINGLE-KERNEL chunked variant.
//
// Modeled after oneCCL's allgatherv_large_sycl_ring_chunking_single_kernel:
// the entire ring (all world_size-1 hops) runs inside ONE kernel launch.
// Cross-rank ordering is performed on-device with per-step signal pads, so
// there are no per-step host launches and no host-side barriers.  Data moves
// by PUSH: each rank writes into its right neighbor's buffer (remote stores
// are far cheaper than remote loads on Xe-Link / PCIe).
//
// Chunking / parallelism model
// ----------------------------
// The per-rank shard of `chunk` elements is split into `num_wg` contiguous
// slices, one per work-group.  Each work-group runs a *fully independent*
// ring pipeline over its own slice:
//   - it only ever waits on the SAME work-group index of the LEFT peer, and
//   - it only signals the SAME work-group index of the RIGHT peer.
// Because work-groups on the same rank never wait on each other, the kernel
// cannot deadlock even if the grid is larger than the number of resident
// work-groups: every work-group makes forward progress independently and
// only blocks on remote peers, which themselves keep retiring work-groups.
//
// Topology (push based): right = (rank + 1) % ws.  Each rank WRITES into the
// RIGHT peer's buffer (remote stores are much cheaper than remote loads on
// Xe-Link / PCIe), matching the oneCCL reference.
//   - Phase 0: copy local shard slice into our OWN output slot `rank`, and
//     also push it into the RIGHT peer's output slot `rank`; signal right
//     (slot 0).
//   - Step t (1..ws-1): wait our OWN signal for phase t-1 (the left peer just
//     pushed block idx=(rank-t+ws)%ws into our output), then forward that
//     block by pushing our output slot idx into the RIGHT peer's output slot
//     idx.  Signal right (slot t).  The final step (t==ws-1) only waits to
//     confirm the last block arrived; it has no block left to forward.
//
// Signal-pad layout (per rank): a flat uint32 array indexed by
//   slot(phase, wg) = phase * num_wg + wg
// `iteration` is a strictly-increasing tag so pads never need clearing
// mid-run (the Python wrapper zeroes them once + barriers before each call).
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

// Publish `val` to a peer pad slot with a system-scope release store.
// atomic_ref is unavailable here, so we order prior writes with a system-scope
// release fence and perform the publish as a naturally-atomic 4-byte volatile
// store.
inline void store_release_sys(uint32_t* addr, uint32_t val) {
  sycl::atomic_fence(
      sycl::memory_order::release, sycl::memory_scope::system);
  *static_cast<volatile uint32_t*>(addr) = val;
}

// Spin until *addr == val, then order subsequent reads with a system-scope
// acquire fence (the load is a naturally-atomic 4-byte volatile read).
inline void wait_eq_sys(uint32_t* addr, uint32_t val) {
  volatile uint32_t* p = static_cast<volatile uint32_t*>(addr);
  while (*p != val) {
  }
  sycl::atomic_fence(
      sycl::memory_order::acquire, sycl::memory_scope::system);
}

template <typename scalar_t, int VEC_SIZE>
struct RingAllgatherSingleKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_shard_ptr;
  const int64_t* rank_buffers_ptr;
  const int64_t* signal_pads_ptr;
  scalar_t* output_ptr;
  int64_t chunk;
  int64_t elems_per_wg;
  int32_t rank;
  int32_t world_size;
  int32_t left;
  int32_t right;
  int32_t num_wg;
  uint32_t tag;

  // Coalesced cooperative copy of `n` elements by one work-group.
  // Uses a vectorized path when both pointers are VEC-aligned.
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

  // Cooperative copy of `n` elements from `src` into TWO destinations in a
  // single read pass (used by phase 0 to write our shard into our own output
  // and push it to the right peer without reading `src` twice).
  inline void wg_copy2(
      const scalar_t* src,
      scalar_t* dst0,
      scalar_t* dst1,
      int64_t n,
      int32_t lid,
      int32_t lsize) const {
    if (n <= 0) return;
    const uintptr_t a = reinterpret_cast<uintptr_t>(src) |
        reinterpret_cast<uintptr_t>(dst0) | reinterpret_cast<uintptr_t>(dst1);
    if ((a % (VEC_SIZE * sizeof(scalar_t))) == 0 && n >= VEC_SIZE) {
      const int64_t nv = n / VEC_SIZE;
      auto sv = reinterpret_cast<const vec_t*>(src);
      auto dv0 = reinterpret_cast<vec_t*>(dst0);
      auto dv1 = reinterpret_cast<vec_t*>(dst1);
      for (int64_t i = lid; i < nv; i += lsize) {
        const vec_t v = sv[i];
        dv0[i] = v;
        dv1[i] = v;
      }
      for (int64_t i = nv * VEC_SIZE + lid; i < n; i += lsize) {
        const scalar_t v = src[i];
        dst0[i] = v;
        dst1[i] = v;
      }
    } else {
      for (int64_t i = lid; i < n; i += lsize) {
        const scalar_t v = src[i];
        dst0[i] = v;
        dst1[i] = v;
      }
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    // This work-group's slice of a single chunk.
    const int64_t base = static_cast<int64_t>(wg) * elems_per_wg;
    int64_t cnt = chunk - base;
    if (cnt > elems_per_wg) cnt = elems_per_wg;
    if (cnt < 0) cnt = 0;

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[rank]);
    uint32_t* right_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[right]);
    scalar_t* right_out =
        reinterpret_cast<scalar_t*>(rank_buffers_ptr[right]);

    // Phase 0: publish our own shard slice (block `rank`).
    {
      const scalar_t* src = input_shard_ptr + base;
      const int64_t slot = static_cast<int64_t>(rank) * chunk + base;
      // Single read pass: place into our own output slot `rank` AND push the
      // same data into the RIGHT peer's output slot `rank`.
      wg_copy2(src, output_ptr + slot, right_out + slot, cnt, lid, lsize);
      // Make the remote write visible to the right peer before signaling.
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0 && world_size > 1) {
        store_release_sys(right_pad + (0 * num_wg + wg), tag);
      }
    }

    // Steps 1..ws-1: forward one freshly-received block per step to the right.
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      // All work-items acquire the left peer's published writes into our output.
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      // The block the left peer just delivered into our output buffer.
      const int32_t idx = (rank - t + world_size) % world_size;

      // The last received block is final for us; no further forwarding.
      if (t < world_size - 1) {
        const scalar_t* src = output_ptr + static_cast<int64_t>(idx) * chunk + base;
        scalar_t* dst = right_out + static_cast<int64_t>(idx) * chunk + base;
        wg_copy(src, dst, cnt, lid, lsize);
        sycl::atomic_fence(
            sycl::memory_order::release, sycl::memory_scope::system);
        item.barrier(sycl::access::fence_space::local_space);
        if (lid == 0) {
          store_release_sys(right_pad + (t * num_wg + wg), tag);
        }
      }
    }
  }
};

// Deterministic (rank-independent) work-group count for a given chunk size.
// Must be identical on every rank because the signal-pad slot layout depends
// on it.
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

// output is rank's slice of symmetric memory and must equal
// rank_buffers_ptr[rank]; peers read it directly.
at::Tensor ring_allgather(
    const at::Tensor& input_shard,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& signal_pads_ptr,
    at::Tensor output,
    int64_t rank,
    int64_t world_size,
    int64_t iteration) {
  TORCH_CHECK(input_shard.dim() == 1, "ring_allgather: input_shard must be 1D");
  TORCH_CHECK(
      input_shard.is_contiguous(), "ring_allgather: input_shard must be contiguous");
  TORCH_CHECK(output.dim() == 1, "ring_allgather: output must be 1D");
  TORCH_CHECK(output.is_contiguous(), "ring_allgather: output must be contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "ring_allgather: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "ring_allgather: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      signal_pads_ptr.dim() == 1 && signal_pads_ptr.size(0) == world_size,
      "ring_allgather: signal_pads_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      signal_pads_ptr.scalar_type() == at::kLong,
      "ring_allgather: signal_pads_ptr must be int64");
  TORCH_CHECK(
      rank >= 0 && rank < world_size,
      "ring_allgather: rank must be in [0, world_size)");
  TORCH_CHECK(
      input_shard.scalar_type() == output.scalar_type(),
      "ring_allgather: input_shard and output must have same dtype");
  TORCH_CHECK(iteration > 0, "ring_allgather: iteration must be > 0");

  const int64_t chunk = input_shard.numel();
  TORCH_CHECK(
      output.numel() == chunk * world_size,
      "ring_allgather: output.numel() must equal input_shard.numel() * world_size");

  if (chunk == 0 || world_size == 1) {
    if (chunk != 0) {
      // world_size == 1: just copy local shard into the single slot.
      output.narrow(0, 0, chunk).copy_(input_shard);
    }
    return output;
  }

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

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

  AT_DISPATCH_FLOAT_AND_BFLOAT16(output.scalar_type(), "ring_allgather", [&]() {
    auto kfn = RingAllgatherSingleKernel<scalar_t, VEC_SIZE>{
        input_shard.data_ptr<scalar_t>(),
        rank_buffers_ptr.data_ptr<int64_t>(),
        signal_pads_ptr.data_ptr<int64_t>(),
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
      "ring_allgather(Tensor input_shard, Tensor rank_buffers_ptr, "
      "Tensor signal_pads_ptr, Tensor(a!) output, int rank, int world_size, "
      "int iteration) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_allgather", ring_allgather);
}
