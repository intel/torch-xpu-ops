#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

// ===========================================================================
// Fused ring allgather + MoE permute  --  SINGLE-KERNEL, PUSH based.
//
// This fuses two stages that a TP+EP dispatch normally runs back to back:
//   1. A pipelined ring allgather that assembles every rank's token shard
//      ([num_tokens_per_rank, hidden]) into the local symmetric `gather`
//      buffer laid out as [world_size, num_tokens_per_rank, hidden].
//   2. A local MoE permute that, for every gathered token, scatters the
//      hidden vector into this rank's expert-grouped `remap` buffer for each
//      top-k slot whose expert is OWNED by this rank (same routing as
//      EpDispatch / LocalPermuteCopy).
//
// Overlap strategy
// ----------------
// The ring forward (a remote PUSH into the right peer plus the signal) is on
// the cross-rank critical path, while the permute is terminal LOCAL work that
// nobody downstream waits on.  So for every block we FORWARD + SIGNAL FIRST,
// then permute that block.  With many resident work-groups the permute of one
// block naturally fills the bubble while other work-groups wait on the next
// hop (work-group level TLP), so the permute is hidden behind communication.
//
// Chunking model (identical to RingAllgather, but TOKEN aligned)
// --------------------------------------------------------------
// chunk = num_tokens_per_rank * hidden.  The chunk is split into `num_wg`
// TOKEN-aligned slices (each work-group owns a contiguous run of whole
// tokens), so the same slice boundaries work for both the byte-wise ring copy
// and the per-token permute.  Each work-group only ever waits on the SAME
// work-group index of the LEFT peer and signals the SAME index of the RIGHT
// peer (see RingAllgather.cpp for the deadlock-freedom argument).
//
// Topology (push): right = (rank + 1) % ws; each rank WRITES into the right
// peer's gather buffer.  Signal-pad slot(phase, wg) = phase * num_wg + wg.
// `iteration` is a strictly increasing tag (pads zeroed once + barriered by
// the Python wrapper before each call).
// ===========================================================================

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

namespace {

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

template <typename scalar_t, int VEC_SIZE>
struct RingAllgatherPermuteSingleKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_shard_ptr;   // [num_tokens_per_rank, hidden]
  const int64_t* rank_buffers_ptr;   // symm gather buffers, one per rank
  const int64_t* signal_pads_ptr;    // signal pads, one per rank
  scalar_t* gather_ptr;              // == rank_buffers_ptr[rank]
  scalar_t* remap_ptr;              // [remap_rows, hidden] (local)
  const int32_t* topk_idx_ptr;     // [world_size * num_tokens_per_rank, topk]
  const int32_t* scatter_idx_ptr;  // [world_size * num_tokens_per_rank, topk]
  int64_t hidden;
  int64_t num_tokens_per_rank;
  int64_t tokens_per_wg;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t num_wg;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;
  uint32_t tag;

  // Coalesced cooperative copy of `n` elements by one work-group.
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

  // Single-read dual-write cooperative copy (phase 0: own gather slot + push).
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

  // Permute every token in this work-group's slice of gathered block `block`
  // into the expert-grouped `remap` buffer.  For each top-k slot whose expert
  // is owned by this rank, the token's hidden vector is copied to
  // remap[scatter_idx[gt, k]].  All threads share the (uniform) token / k /
  // ownership control flow and cooperate on the per-token hidden copy.
  inline void wg_permute_block(
      int32_t block,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t gt =
          static_cast<int64_t>(block) * num_tokens_per_rank + local_t;
      const scalar_t* src = gather_ptr + gt * hidden;
      const int64_t topk_base = gt * topk;
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t expert = topk_idx_ptr[topk_base + k];
        int32_t owner;
        if (expert < boundary) {
          owner = expert / (base_experts + 1);
        } else {
          owner = rem_experts + (expert - boundary) / base_experts;
        }
        if (owner != rank) continue;
        const int32_t dst_row = scatter_idx_ptr[topk_base + k];
        scalar_t* dst = remap_ptr + static_cast<int64_t>(dst_row) * hidden;
        wg_copy(src, dst, hidden, lid, lsize);
      }
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    // This work-group's TOKEN-aligned slice.
    const int64_t token_base = static_cast<int64_t>(wg) * tokens_per_wg;
    int64_t token_cnt = num_tokens_per_rank - token_base;
    if (token_cnt > tokens_per_wg) token_cnt = tokens_per_wg;
    if (token_cnt < 0) token_cnt = 0;

    const int64_t chunk = num_tokens_per_rank * hidden;
    const int64_t base = token_base * hidden;  // flat element offset
    const int64_t cnt = token_cnt * hidden;    // flat element count

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[rank]);
    uint32_t* right_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[right]);
    scalar_t* right_gather =
        reinterpret_cast<scalar_t*>(rank_buffers_ptr[right]);

    // Phase 0: publish our own shard (block `rank`) then permute it.
    {
      const scalar_t* src = input_shard_ptr + base;
      const int64_t slot = static_cast<int64_t>(rank) * chunk + base;
      wg_copy2(src, gather_ptr + slot, right_gather + slot, cnt, lid, lsize);
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0 && world_size > 1) {
        store_release_sys(right_pad + (0 * num_wg + wg), tag);
      }
      // Terminal local permute of block `rank` (overlaps peers' next hop).
      wg_permute_block(rank, token_base, token_cnt, lid, lsize);
    }

    // Steps 1..ws-1: forward + signal FIRST, then permute the received block.
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      const int32_t idx = (rank - t + world_size) % world_size;

      if (t < world_size - 1) {
        const scalar_t* src =
            gather_ptr + static_cast<int64_t>(idx) * chunk + base;
        scalar_t* dst = right_gather + static_cast<int64_t>(idx) * chunk + base;
        wg_copy(src, dst, cnt, lid, lsize);
        sycl::atomic_fence(
            sycl::memory_order::release, sycl::memory_scope::system);
        item.barrier(sycl::access::fence_space::local_space);
        if (lid == 0) {
          store_release_sys(right_pad + (t * num_wg + wg), tag);
        }
      }
      // Permute this block's tokens (terminal local work for all blocks).
      wg_permute_block(idx, token_base, token_cnt, lid, lsize);
    }
  }
};

// Deterministic (rank-independent) TOKEN-aligned work-group split.
inline void compute_launch_tokens(
    int64_t num_tokens,
    int64_t hidden,
    int64_t threads,
    int VEC_SIZE,
    int32_t& num_wg,
    int64_t& tokens_per_wg) {
  const int64_t chunk = num_tokens * hidden;
  const int64_t per_wg = threads * VEC_SIZE;
  int64_t nwg = (chunk + per_wg - 1) / per_wg;
  if (nwg < 1) nwg = 1;
  if (nwg > RING_MAX_WG) nwg = RING_MAX_WG;
  int64_t tpw = (num_tokens + nwg - 1) / nwg;
  if (tpw < 1) tpw = 1;
  nwg = (num_tokens + tpw - 1) / tpw;
  if (nwg < 1) nwg = 1;
  num_wg = static_cast<int32_t>(nwg);
  tokens_per_wg = tpw;
}

}  // namespace

// `gather_output` is rank's slice of symmetric memory and must equal
// rank_buffers_ptr[rank]; peers push into it directly.  `remap_output` is a
// local (non-symmetric) expert-grouped buffer this rank owns.
at::Tensor ring_allgather_permute(
    const at::Tensor& input_shard,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& signal_pads_ptr,
    at::Tensor gather_output,
    at::Tensor remap_output,
    const at::Tensor& topk_idx,
    const at::Tensor& scatter_idx,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size,
    int64_t iteration) {
  TORCH_CHECK(input_shard.dim() == 2, "ring_allgather_permute: input_shard must be 2D [tokens, hidden]");
  TORCH_CHECK(input_shard.is_contiguous(), "ring_allgather_permute: input_shard must be contiguous");
  TORCH_CHECK(gather_output.dim() == 1, "ring_allgather_permute: gather_output must be 1D");
  TORCH_CHECK(gather_output.is_contiguous(), "ring_allgather_permute: gather_output must be contiguous");
  TORCH_CHECK(remap_output.dim() == 2, "ring_allgather_permute: remap_output must be 2D [rows, hidden]");
  TORCH_CHECK(remap_output.is_contiguous(), "ring_allgather_permute: remap_output must be contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "ring_allgather_permute: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "ring_allgather_permute: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      signal_pads_ptr.dim() == 1 && signal_pads_ptr.size(0) == world_size,
      "ring_allgather_permute: signal_pads_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      signal_pads_ptr.scalar_type() == at::kLong,
      "ring_allgather_permute: signal_pads_ptr must be int64");
  TORCH_CHECK(topk_idx.dim() == 2, "ring_allgather_permute: topk_idx must be 2D");
  TORCH_CHECK(topk_idx.scalar_type() == at::kInt, "ring_allgather_permute: topk_idx must be int32");
  TORCH_CHECK(topk_idx.is_contiguous());
  TORCH_CHECK(
      scatter_idx.dim() == 2 && scatter_idx.sizes() == topk_idx.sizes(),
      "ring_allgather_permute: scatter_idx must match topk_idx shape");
  TORCH_CHECK(scatter_idx.scalar_type() == at::kInt, "ring_allgather_permute: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(rank >= 0 && rank < world_size, "ring_allgather_permute: rank out of range");
  TORCH_CHECK(
      input_shard.scalar_type() == gather_output.scalar_type() &&
          input_shard.scalar_type() == remap_output.scalar_type(),
      "ring_allgather_permute: input/gather/remap must have same dtype");
  TORCH_CHECK(iteration > 0, "ring_allgather_permute: iteration must be > 0");

  const int64_t num_tokens_per_rank = input_shard.size(0);
  const int64_t hidden = input_shard.size(1);
  const int64_t chunk = num_tokens_per_rank * hidden;

  TORCH_CHECK(
      gather_output.numel() == chunk * world_size,
      "ring_allgather_permute: gather_output.numel() must equal tokens*hidden*world_size");
  TORCH_CHECK(
      remap_output.size(1) == hidden,
      "ring_allgather_permute: remap_output hidden must match input_shard");
  TORCH_CHECK(
      topk_idx.size(0) == num_tokens_per_rank * world_size,
      "ring_allgather_permute: topk_idx first dim must equal world_size * tokens");

  if (chunk == 0) {
    return remap_output;
  }

  c10::Device device(c10::DeviceType::XPU, remap_output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  const int32_t ws = static_cast<int32_t>(world_size);
  const int32_t r = static_cast<int32_t>(rank);
  const int32_t right = (r + 1) % ws;
  const int32_t topk = static_cast<int32_t>(topk_idx.size(1));
  const uint32_t tag = static_cast<uint32_t>(iteration);

  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;
  int32_t num_wg = 1;
  int64_t tokens_per_wg = num_tokens_per_rank;
  compute_launch_tokens(
      num_tokens_per_rank, hidden, threads, VEC_SIZE, num_wg, tokens_per_wg);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      remap_output.scalar_type(), "ring_allgather_permute", [&]() {
        auto kfn = RingAllgatherPermuteSingleKernel<scalar_t, VEC_SIZE>{
            input_shard.data_ptr<scalar_t>(),
            rank_buffers_ptr.data_ptr<int64_t>(),
            signal_pads_ptr.data_ptr<int64_t>(),
            gather_output.data_ptr<scalar_t>(),
            remap_output.data_ptr<scalar_t>(),
            topk_idx.data_ptr<int32_t>(),
            scatter_idx.data_ptr<int32_t>(),
            hidden,
            num_tokens_per_rank,
            tokens_per_wg,
            topk,
            r,
            ws,
            right,
            num_wg,
            base_experts,
            rem_experts,
            boundary,
            tag};
        sycl_kernel_submit(
            sycl::range<1>(static_cast<size_t>(num_wg) * threads),
            sycl::range<1>(threads),
            queue,
            kfn);
      });

  return remap_output;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ring_allgather_permute(Tensor input_shard, Tensor rank_buffers_ptr, "
      "Tensor signal_pads_ptr, Tensor(a!) gather_output, Tensor(b!) remap_output, "
      "Tensor topk_idx, Tensor scatter_idx, int num_experts, int rank, "
      "int world_size, int iteration) -> Tensor(b!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_allgather_permute", ring_allgather_permute);
}
