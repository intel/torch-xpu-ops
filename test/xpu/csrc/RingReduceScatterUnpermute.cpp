#include <cstdlib>
#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

// ===========================================================================
// Fused MoE unpermute + ring reduce-scatter  --  SINGLE-KERNEL, PUSH based.
//
// This fuses two stages of a TP MoE combine:
//   1. A local MoE unpermute that, for every global token, gathers from
//      expert_output using absolute scatter_idx and forms the weighted
//      partial:  contrib[token, h] = sum_k topk_weights[token, k]
//                    * expert_output[scatter_idx[token, k], h]
//   2. A pipelined ring reduce-scatter that SUMS those per-rank partials so
//      that rank b ends up with the fully combined tokens of block b.
//
// Unlike a precomputed `input` tensor, the partial for each block is produced
// ON THE FLY by the unpermute right where the ring needs it.  The unpermute
// reads only LOCAL expert_output, so while one work-group spins waiting for
// the incoming remote partial, other resident work-groups run their unpermute
// compute (work-group level TLP) -- the local compute is hidden behind the
// cross-rank wait.
//
// Topology (push): right = (rank + 1) % ws; the running partial for final
// block b travels b+1 -> ... -> b, gaining one rank's contribution per hop.
// From rank r's view, at step t it handles block b_t = (rank-1-t+ws)%ws:
//   - Phase 0 (t=0): push our own unpermuted contrib for b0=(rank-1+ws)%ws
//     into the right peer's acc[b0]; signal right (slot 0).
//   - Step t (1..ws-2): wait phase t-1 (left pushed the partial for b_t into
//     our acc[b_t]); push unpermute(b_t) + acc[b_t] into right.acc[b_t];
//     signal right (slot t).
//   - Step ws-1: b_t == rank is final; write unpermute(rank) + acc[rank] into
//     `output` (no push).
//
// chunk = num_tokens_per_rank * hidden; TOKEN-aligned work-group slices.
// Signal-pad slot(phase, wg) = phase * num_wg + wg; `iteration` is a strictly
// increasing tag (pads zeroed once + barriered by the Python wrapper).
// ===========================================================================

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// The bandwidth-critical 16-byte store here writes the reduced partial directly
// into the RIGHT peer's acc buffer, which that peer then remote-reads (after a
// system-scope acquire fence) on the next ring step. An Intel GPU LSC store with
// L1WB_L3WB (writeback) cache control leaves the data in the writeback cache; the
// system-scope `release` fence orders it but does NOT reliably drain it to the
// shared coherence point where the downstream peer's load observes it in time.
// The result is a correctness bug: the peer reads a STALE acc block and folds it
// into the reduce-scatter, producing garbage combine outputs. (Empirically this
// manifests whenever the live token count differs from the buffer's
// num_max_tokens_per_rank, i.e. every real decode/prefill step.) The peer write
// therefore defaults to a plain (coherent) store; the ~18% writeback win on B60
// is not worth the multi-hop staleness hazard.
// Opt-in (only safe on fabrics where the writeback drains before the peer read):
// RING_LSC_STORE_WB. Opt-out of the LSC path entirely: RING_NO_LSC_STORE.
#if !defined(RING_LSC_STORE) && !defined(RING_NO_LSC_STORE)
#define RING_LSC_STORE 1
#endif
#if defined(RING_LSC_STORE) && !defined(RING_LSC_STORE_WB)
#define RING_LSC_STORE_PLAIN 1
#endif

#if defined(RING_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
typedef uint32_t ring_lsc_u4 __attribute__((ext_vector_type(4)));
enum RingLscStcc { RING_LSC_STCC_L1WB_L3WB = 7 };
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_uint4(
    __attribute__((opencl_global)) ring_lsc_u4* base,
    int off,
    ring_lsc_u4 val,
    enum RingLscStcc cc);
#endif

namespace {

constexpr int32_t RING_MAX_WG = 256;

// 16-byte vector store with optional LSC cache control (see RING_LSC_STORE).
template <typename vec_t>
inline void ring_vec_store(vec_t* dst, vec_t vd) {
#if defined(RING_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  if constexpr (sizeof(vec_t) == 16) {
#if defined(RING_LSC_STORE_PLAIN)
    // Plain coherent store: the release fence drains it to the coherence point
    // so the downstream peer's acquire-load observes it (see note above).
    *dst = vd;
#else
    ring_lsc_u4 v = *reinterpret_cast<ring_lsc_u4*>(&vd);
    __builtin_IB_lsc_store_global_uint4(
        (__attribute__((opencl_global)) ring_lsc_u4*)(dst),
        0,
        v,
        RING_LSC_STCC_L1WB_L3WB);
#endif
  } else {
    *dst = vd;
  }
#else
  *dst = vd;
#endif
}

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
struct RingReduceScatterUnpermuteSingleKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* expert_output_ptr;  // [num_tokens*topk, hidden] (local, all experts)
  const int64_t* rank_buffers_ptr;    // symm acc buffers, one per rank
  const int64_t* signal_pads_ptr;     // signal pads, one per rank
  scalar_t* acc_ptr;                  // == rank_buffers_ptr[rank]
  scalar_t* output_ptr;              // [num_tokens_per_rank, hidden] (local)
  const int32_t* scatter_idx_ptr;    // [world_size * tokens, topk] (absolute)
  const float* topk_weights_ptr;     // [world_size * tokens, topk]
  int64_t hidden;
  int64_t num_tokens_per_rank;
  int64_t tokens_per_wg;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t num_wg;
  uint32_t tag;

  // Unpermute one token: form the weighted sum of expert outputs for all topk
  // slots using absolute scatter_idx, optionally add the incoming partial
  // `acc_row`, and store into `dst_row`.
  inline void unpermute_token(
      int64_t gt,
      const scalar_t* acc_row,  // nullptr for phase 0 (no incoming partial)
      scalar_t* dst_row,
      int32_t lid,
      int32_t lsize) const {
    const int64_t hidden_vecs = hidden / VEC_SIZE;
    const int64_t topk_base = gt * topk;
    for (int64_t vh = lid; vh < hidden_vecs; vh += lsize) {
      const int64_t h_start = vh * VEC_SIZE;
      float acc[VEC_SIZE];
      if (acc_row != nullptr) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
          acc[i] = static_cast<float>(acc_row[h_start + i]);
      } else {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
          acc[i] = 0.0f;
      }
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        if (src_row < 0) continue;
        const float w = topk_weights_ptr[topk_base + k];
        const scalar_t* src =
            expert_output_ptr + static_cast<int64_t>(src_row) * hidden + h_start;
        // Single coalesced VEC_SIZE-wide load (h_start is VEC_SIZE-aligned and
        // hidden % VEC_SIZE == 0, so `src` is naturally aligned). This replaces
        // VEC_SIZE scalar loads with one wide load per source row -- the gather
        // is the HBM-bandwidth-critical path, so wide loads roughly double its
        // achieved bandwidth vs per-element reads.
        const vec_t sv = *reinterpret_cast<const vec_t*>(src);
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          const vec_elem_t bits = sv[i];
          acc[i] += w *
              static_cast<float>(*reinterpret_cast<const scalar_t*>(&bits));
        }
      }
      // Single coalesced VEC_SIZE-wide store; when dst_row is the remote peer's
      // acc this is one PCIe block transaction instead of VEC_SIZE per-element
      // writes (the bandwidth-critical path).
      vec_t vd;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        const scalar_t s = static_cast<scalar_t>(acc[i]);
        vd[i] = *reinterpret_cast<const vec_elem_t*>(&s);
      }
      ring_vec_store(reinterpret_cast<vec_t*>(dst_row + h_start), vd);
    }
    // Tail for hidden not divisible by VEC_SIZE.
    for (int64_t h = hidden_vecs * VEC_SIZE + lid; h < hidden; h += lsize) {
      float a = (acc_row != nullptr) ? static_cast<float>(acc_row[h]) : 0.0f;
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        if (src_row < 0) continue;
        const float w = topk_weights_ptr[topk_base + k];
        a += w *
            static_cast<float>(
                 expert_output_ptr[static_cast<int64_t>(src_row) * hidden + h]);
      }
      dst_row[h] = static_cast<scalar_t>(a);
    }
  }

  // Unpermute (+ optional add of acc) for every token in this WG's slice of
  // block `block`, writing into `dst` (the right peer's acc, or our output).
  inline void wg_unpermute_block(
      int32_t block,
      const scalar_t* acc_base,  // nullptr for phase 0
      scalar_t* dst_base,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t gt =
          static_cast<int64_t>(block) * num_tokens_per_rank + local_t;
      const int64_t row_off =
          (static_cast<int64_t>(block) * num_tokens_per_rank + local_t) * hidden;
      const scalar_t* acc_row =
          (acc_base != nullptr) ? acc_base + row_off : nullptr;
      scalar_t* dst_row = dst_base + row_off;
      unpermute_token(gt, acc_row, dst_row, lid, lsize);
    }
  }

  // Final block writes into the compact `output` ([tokens, hidden]); the row
  // offset there is local-token relative (no block dimension).
  inline void wg_unpermute_final(
      int32_t block,
      const scalar_t* acc_base,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t gt =
          static_cast<int64_t>(block) * num_tokens_per_rank + local_t;
      const int64_t acc_off =
          (static_cast<int64_t>(block) * num_tokens_per_rank + local_t) * hidden;
      const scalar_t* acc_row = acc_base + acc_off;
      scalar_t* dst_row = output_ptr + local_t * hidden;
      unpermute_token(gt, acc_row, dst_row, lid, lsize);
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    const int64_t token_base = static_cast<int64_t>(wg) * tokens_per_wg;
    int64_t token_cnt = num_tokens_per_rank - token_base;
    if (token_cnt > tokens_per_wg) token_cnt = tokens_per_wg;
    if (token_cnt < 0) token_cnt = 0;

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[rank]);
    uint32_t* right_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[right]);
    scalar_t* right_acc =
        reinterpret_cast<scalar_t*>(rank_buffers_ptr[right]);

    // Phase 0: seed the partial for block b0 with our own contribution.
    {
      const int32_t b0 = (rank - 1 + world_size) % world_size;
      wg_unpermute_block(
          b0, nullptr, right_acc, token_base, token_cnt, lid, lsize);
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0) {
        store_release_sys(right_pad + (0 * num_wg + wg), tag);
      }
    }

    // Steps 1..ws-1: fold our contribution into the incoming partial.
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      const int32_t b_t = (rank - 1 - t + 2 * world_size) % world_size;

      if (t < world_size - 1) {
        // right.acc[b_t] = unpermute(b_t) + acc[b_t].
        wg_unpermute_block(
            b_t, acc_ptr, right_acc, token_base, token_cnt, lid, lsize);
        sycl::atomic_fence(
            sycl::memory_order::release, sycl::memory_scope::system);
        item.barrier(sycl::access::fence_space::local_space);
        if (lid == 0) {
          store_release_sys(right_pad + (t * num_wg + wg), tag);
        }
      } else {
        // Final: block `rank` (b_t == rank) is fully combined here.
        wg_unpermute_final(b_t, acc_ptr, token_base, token_cnt, lid, lsize);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Owner-based (Expert-Parallel) variant.
//
// Identical ring reduce-scatter topology and signalling as the kernel above,
// but the fused unpermute is OWNERSHIP-FILTERED: for each token it accumulates
// only the top-k slots whose expert is owned by this rank (contiguous owner
// partition of `num_experts` across `world_size`).  Each rank therefore reads
// ~topk/world_size expert_output rows per token instead of all topk, cutting
// the HBM-bandwidth-critical scattered gather by ~world_size x.  Correctness is
// unchanged: every expert is owned by exactly one rank, so summing each rank's
// owned partial around the ring reconstructs the full weighted combine.
// ---------------------------------------------------------------------------
template <typename scalar_t, int VEC_SIZE>
struct RingReduceScatterUnpermuteEpKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* expert_output_ptr;
  const int64_t* rank_buffers_ptr;
  const int64_t* signal_pads_ptr;
  scalar_t* acc_ptr;
  scalar_t* output_ptr;
  const int32_t* scatter_idx_ptr;
  const int32_t* topk_idx_ptr;       // [world_size*tokens, topk] expert ids
  const float* topk_weights_ptr;
  int64_t hidden;
  int64_t num_tokens_per_rank;
  int64_t tokens_per_wg;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t num_wg;
  uint32_t tag;
  int32_t base_experts;              // num_experts / world_size
  int32_t rem_experts;               // num_experts % world_size
  int32_t boundary;                  // rem_experts * (base_experts + 1)

  inline bool owns(int32_t expert) const {
    int32_t owner;
    if (expert < boundary) {
      owner = expert / (base_experts + 1);
    } else {
      owner = rem_experts + (expert - boundary) / base_experts;
    }
    return owner == rank;
  }

  // Resolve, ONCE per token, which top-k slots this rank must actually gather:
  // a slot counts only if its expert is owned by this rank AND its scatter row
  // is valid (>= 0).  The result is a bitmask over the (small) top-k slots.
  // Two wins over checking ownership inside the hidden loop:
  //   1. The ownership integer-divisions run topk times per token instead of
  //      hidden_vecs*topk times (they don't depend on the hidden position).
  //   2. mask == 0 means this token contributes nothing on this rank, so we can
  //      skip the bandwidth-critical scattered HBM gather entirely and just
  //      pass the incoming partial through (see the fast path below).
  inline uint32_t resolve_owned_mask(int64_t topk_base) const {
    uint32_t mask = 0;
    for (int32_t k = 0; k < topk; ++k) {
      if (!owns(topk_idx_ptr[topk_base + k]))
        continue;
      if (scatter_idx_ptr[topk_base + k] < 0)
        continue;
      mask |= (1u << k);
    }
    return mask;
  }

  inline void unpermute_token(
      int64_t gt,
      const scalar_t* acc_row,
      scalar_t* dst_row,
      int32_t lid,
      int32_t lsize) const {
    const int64_t hidden_vecs = hidden / VEC_SIZE;
    const int64_t topk_base = gt * topk;
    const uint32_t owned_mask = resolve_owned_mask(topk_base);

    // Fast path: this rank owns none of the token's experts, so its partial is
    // unchanged.  Copy the incoming acc straight to dst (or write zero when
    // seeding in phase 0) -- no scattered expert_output gather at all.
    if (owned_mask == 0) {
      for (int64_t vh = lid; vh < hidden_vecs; vh += lsize) {
        const int64_t h_start = vh * VEC_SIZE;
        vec_t vd;
        if (acc_row != nullptr) {
          vd = *reinterpret_cast<const vec_t*>(acc_row + h_start);
        } else {
#pragma unroll
          for (int i = 0; i < VEC_SIZE; ++i)
            vd[i] = vec_elem_t(0);
        }
        ring_vec_store(reinterpret_cast<vec_t*>(dst_row + h_start), vd);
      }
      for (int64_t h = hidden_vecs * VEC_SIZE + lid; h < hidden; h += lsize) {
        dst_row[h] =
            (acc_row != nullptr) ? acc_row[h] : static_cast<scalar_t>(0);
      }
      return;
    }

    for (int64_t vh = lid; vh < hidden_vecs; vh += lsize) {
      const int64_t h_start = vh * VEC_SIZE;
      float acc[VEC_SIZE];
      if (acc_row != nullptr) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
          acc[i] = static_cast<float>(acc_row[h_start + i]);
      } else {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
          acc[i] = 0.0f;
      }
      for (int32_t k = 0; k < topk; ++k) {
        // Ownership + validity already resolved into owned_mask (no division,
        // no negative-row check here on the hidden-bandwidth-critical path).
        if (!((owned_mask >> k) & 1u)) continue;
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        const float w = topk_weights_ptr[topk_base + k];
        const scalar_t* src =
            expert_output_ptr + static_cast<int64_t>(src_row) * hidden + h_start;
        const vec_t sv = *reinterpret_cast<const vec_t*>(src);
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          const vec_elem_t bits = sv[i];
          acc[i] += w *
              static_cast<float>(*reinterpret_cast<const scalar_t*>(&bits));
        }
      }
      vec_t vd;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        const scalar_t s = static_cast<scalar_t>(acc[i]);
        vd[i] = *reinterpret_cast<const vec_elem_t*>(&s);
      }
      ring_vec_store(reinterpret_cast<vec_t*>(dst_row + h_start), vd);
    }
    for (int64_t h = hidden_vecs * VEC_SIZE + lid; h < hidden; h += lsize) {
      float a = (acc_row != nullptr) ? static_cast<float>(acc_row[h]) : 0.0f;
      for (int32_t k = 0; k < topk; ++k) {
        if (!((owned_mask >> k) & 1u)) continue;
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        const float w = topk_weights_ptr[topk_base + k];
        a += w *
            static_cast<float>(
                 expert_output_ptr[static_cast<int64_t>(src_row) * hidden + h]);
      }
      dst_row[h] = static_cast<scalar_t>(a);
    }
  }

  inline void wg_unpermute_block(
      int32_t block,
      const scalar_t* acc_base,
      scalar_t* dst_base,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t gt =
          static_cast<int64_t>(block) * num_tokens_per_rank + local_t;
      const int64_t row_off =
          (static_cast<int64_t>(block) * num_tokens_per_rank + local_t) * hidden;
      const scalar_t* acc_row =
          (acc_base != nullptr) ? acc_base + row_off : nullptr;
      scalar_t* dst_row = dst_base + row_off;
      unpermute_token(gt, acc_row, dst_row, lid, lsize);
    }
  }

  inline void wg_unpermute_final(
      int32_t block,
      const scalar_t* acc_base,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t gt =
          static_cast<int64_t>(block) * num_tokens_per_rank + local_t;
      const int64_t acc_off =
          (static_cast<int64_t>(block) * num_tokens_per_rank + local_t) * hidden;
      const scalar_t* acc_row = acc_base + acc_off;
      scalar_t* dst_row = output_ptr + local_t * hidden;
      unpermute_token(gt, acc_row, dst_row, lid, lsize);
    }
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    const int64_t token_base = static_cast<int64_t>(wg) * tokens_per_wg;
    int64_t token_cnt = num_tokens_per_rank - token_base;
    if (token_cnt > tokens_per_wg) token_cnt = tokens_per_wg;
    if (token_cnt < 0) token_cnt = 0;

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[rank]);
    uint32_t* right_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[right]);
    scalar_t* right_acc =
        reinterpret_cast<scalar_t*>(rank_buffers_ptr[right]);

    {
      const int32_t b0 = (rank - 1 + world_size) % world_size;
      wg_unpermute_block(
          b0, nullptr, right_acc, token_base, token_cnt, lid, lsize);
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0) {
        store_release_sys(right_pad + (0 * num_wg + wg), tag);
      }
    }

    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      const int32_t b_t = (rank - 1 - t + 2 * world_size) % world_size;

      if (t < world_size - 1) {
        wg_unpermute_block(
            b_t, acc_ptr, right_acc, token_base, token_cnt, lid, lsize);
        sycl::atomic_fence(
            sycl::memory_order::release, sycl::memory_scope::system);
        item.barrier(sycl::access::fence_space::local_space);
        if (lid == 0) {
          store_release_sys(right_pad + (t * num_wg + wg), tag);
        }
      } else {
        wg_unpermute_final(b_t, acc_ptr, token_base, token_cnt, lid, lsize);
      }
    }
  }
};

inline int32_t ring_max_wg_runtime() {
  static const int32_t v = [] {
    const char* e = std::getenv("RING_MAX_WG_OVERRIDE");
    if (e) {
      int x = std::atoi(e);
      if (x > 0) return x;
    }
    return static_cast<int>(RING_MAX_WG);
  }();
  return v;
}

// Occupancy-based launch geometry.  The kernel is memory-latency-bound on the
// scattered top-k gather, so we size the grid to ~2x oversubscribe the machine's
// resident-subgroup capacity (eu_count) for latency hiding, then snap
// tokens_per_wg to a power of two so the token stream splits evenly across
// work-groups (uneven splits measurably regress due to tail load imbalance and
// per-wg signal skew).  num_wg is additionally clamped to the token count and to
// the signal-pad capacity (RING_MAX_WG).
inline int64_t next_pow2(int64_t x) {
  int64_t p = 1;
  while (p < x) p <<= 1;
  return p;
}

inline void compute_launch_tokens(
    int64_t num_tokens,
    int64_t /*hidden*/,
    int64_t /*threads*/,
    int /*VEC_SIZE*/,
    int64_t eu_count,
    int64_t num_wg_hint,
    int32_t& num_wg,
    int64_t& tokens_per_wg) {
  const int64_t cap = ring_max_wg_runtime();
  // Autotuner override: a positive hint pins the work-group count (still snapped
  // to an even power-of-two token split and clamped to the pad/token limits).
  int64_t num_wg_target;
  if (num_wg_hint > 0) {
    num_wg_target = num_wg_hint;
  } else {
    // 2x oversubscription of the EU array is the latency-hiding sweet spot for
    // the scattered gather; beyond it per-wg signal overhead dominates.
    num_wg_target = 2 * (eu_count > 0 ? eu_count : 64);
  }
  if (num_wg_target > cap) num_wg_target = cap;
  if (num_wg_target < 1) num_wg_target = 1;

  // Even (power-of-two) tokens per work-group avoids the load-imbalance spikes
  // seen with odd splits (e.g. 11 or 13 tokens/wg).
  int64_t tpw = next_pow2((num_tokens + num_wg_target - 1) / num_wg_target);
  if (tpw < 1) tpw = 1;

  int64_t nwg = (num_tokens + tpw - 1) / tpw;
  if (nwg < 1) nwg = 1;
  if (nwg > cap) {
    nwg = cap;
    tpw = (num_tokens + nwg - 1) / nwg;
    nwg = (num_tokens + tpw - 1) / tpw;
  }
  num_wg = static_cast<int32_t>(nwg);
  tokens_per_wg = tpw;
}

}  // namespace

// `acc` is rank's symmetric-memory buffer and must equal
// rank_buffers_ptr[rank]; peers push into it directly.  `expert_output` and
// `output` are local (non-symmetric) tensors this rank owns.
at::Tensor ring_reduce_scatter_unpermute(
    const at::Tensor& expert_output,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& signal_pads_ptr,
    at::Tensor acc,
    at::Tensor output,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    int64_t rank,
    int64_t world_size,
    int64_t iteration,
    int64_t num_wg_hint) {
  TORCH_CHECK(expert_output.dim() == 2, "ring_reduce_scatter_unpermute: expert_output must be 2D [rows, hidden]");
  TORCH_CHECK(expert_output.is_contiguous(), "ring_reduce_scatter_unpermute: expert_output must be contiguous");
  TORCH_CHECK(acc.dim() == 1, "ring_reduce_scatter_unpermute: acc must be 1D");
  TORCH_CHECK(acc.is_contiguous(), "ring_reduce_scatter_unpermute: acc must be contiguous");
  TORCH_CHECK(output.dim() == 2, "ring_reduce_scatter_unpermute: output must be 2D [tokens, hidden]");
  TORCH_CHECK(output.is_contiguous(), "ring_reduce_scatter_unpermute: output must be contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "ring_reduce_scatter_unpermute: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "ring_reduce_scatter_unpermute: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      signal_pads_ptr.dim() == 1 && signal_pads_ptr.size(0) == world_size,
      "ring_reduce_scatter_unpermute: signal_pads_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      signal_pads_ptr.scalar_type() == at::kLong,
      "ring_reduce_scatter_unpermute: signal_pads_ptr must be int64");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "ring_reduce_scatter_unpermute: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(scatter_idx.scalar_type() == at::kInt, "ring_reduce_scatter_unpermute: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(
      topk_weights.dim() == 2 && topk_weights.sizes() == scatter_idx.sizes(),
      "ring_reduce_scatter_unpermute: topk_weights must match scatter_idx shape");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "ring_reduce_scatter_unpermute: topk_weights must be float32");
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(rank >= 0 && rank < world_size, "ring_reduce_scatter_unpermute: rank out of range");
  TORCH_CHECK(
      expert_output.scalar_type() == output.scalar_type() &&
          expert_output.scalar_type() == acc.scalar_type(),
      "ring_reduce_scatter_unpermute: expert_output/acc/output must have same dtype");
  TORCH_CHECK(iteration > 0, "ring_reduce_scatter_unpermute: iteration must be > 0");

  const int64_t num_tokens_per_rank = output.size(0);
  const int64_t hidden = output.size(1);
  const int64_t chunk = num_tokens_per_rank * hidden;

  TORCH_CHECK(
      expert_output.size(1) == hidden,
      "ring_reduce_scatter_unpermute: expert_output hidden must match output");
  TORCH_CHECK(
      acc.numel() == chunk * world_size,
      "ring_reduce_scatter_unpermute: acc.numel() must equal tokens*hidden*world_size");
  TORCH_CHECK(
      scatter_idx.size(0) == num_tokens_per_rank * world_size,
      "ring_reduce_scatter_unpermute: scatter_idx first dim must equal world_size * tokens");

  if (chunk == 0) {
    return output;
  }

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  const int32_t ws = static_cast<int32_t>(world_size);
  const int32_t r = static_cast<int32_t>(rank);
  const int32_t right = (r + 1) % ws;
  const int32_t topk = static_cast<int32_t>(scatter_idx.size(1));
  const uint32_t tag = static_cast<uint32_t>(iteration);

  constexpr int VEC_SIZE = 8;
  // One work-group covers one hidden row with coalesced VEC_SIZE loads.
  int64_t threads = hidden / VEC_SIZE;
  if (threads < 1) threads = 1;
  if (threads > 1024) threads = 1024;
  const int64_t eu_count = static_cast<int64_t>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());
  int32_t num_wg = 1;
  int64_t tokens_per_wg = num_tokens_per_rank;
  compute_launch_tokens(
      num_tokens_per_rank, hidden, threads, VEC_SIZE, eu_count, num_wg_hint,
      num_wg, tokens_per_wg);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ring_reduce_scatter_unpermute", [&]() {
        auto kfn = RingReduceScatterUnpermuteSingleKernel<scalar_t, VEC_SIZE>{
            expert_output.data_ptr<scalar_t>(),
            rank_buffers_ptr.data_ptr<int64_t>(),
            signal_pads_ptr.data_ptr<int64_t>(),
            acc.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            scatter_idx.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            hidden,
            num_tokens_per_rank,
            tokens_per_wg,
            topk,
            r,
            ws,
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

// Owner-based (EP) variant: same ring reduce-scatter, ownership-filtered
// unpermute.  Extra `topk_idx` (expert ids) + `num_experts` drive the filter.
at::Tensor ring_reduce_scatter_unpermute_ep(
    const at::Tensor& expert_output,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& signal_pads_ptr,
    at::Tensor acc,
    at::Tensor output,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_idx,
    const at::Tensor& topk_weights,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size,
    int64_t iteration,
    int64_t num_wg_hint) {
  TORCH_CHECK(expert_output.dim() == 2, "ep: expert_output must be 2D [rows, hidden]");
  TORCH_CHECK(expert_output.is_contiguous(), "ep: expert_output must be contiguous");
  TORCH_CHECK(acc.dim() == 1 && acc.is_contiguous(), "ep: acc must be 1D contiguous");
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous(), "ep: output must be 2D contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size &&
          rank_buffers_ptr.scalar_type() == at::kLong,
      "ep: rank_buffers_ptr must be int64 1D size world_size");
  TORCH_CHECK(
      signal_pads_ptr.dim() == 1 && signal_pads_ptr.size(0) == world_size &&
          signal_pads_ptr.scalar_type() == at::kLong,
      "ep: signal_pads_ptr must be int64 1D size world_size");
  TORCH_CHECK(scatter_idx.dim() == 2 && scatter_idx.scalar_type() == at::kInt &&
      scatter_idx.is_contiguous(), "ep: scatter_idx must be int32 2D contiguous");
  TORCH_CHECK(topk_idx.dim() == 2 && topk_idx.sizes() == scatter_idx.sizes() &&
      topk_idx.scalar_type() == at::kInt && topk_idx.is_contiguous(),
      "ep: topk_idx must be int32 and match scatter_idx shape");
  TORCH_CHECK(topk_weights.dim() == 2 && topk_weights.sizes() == scatter_idx.sizes() &&
      topk_weights.scalar_type() == at::kFloat && topk_weights.is_contiguous(),
      "ep: topk_weights must be float32 and match scatter_idx shape");
  TORCH_CHECK(rank >= 0 && rank < world_size, "ep: rank out of range");
  TORCH_CHECK(
      expert_output.scalar_type() == output.scalar_type() &&
          expert_output.scalar_type() == acc.scalar_type(),
      "ep: expert_output/acc/output must share dtype");
  TORCH_CHECK(iteration > 0, "ep: iteration must be > 0");

  const int64_t num_tokens_per_rank = output.size(0);
  const int64_t hidden = output.size(1);
  const int64_t chunk = num_tokens_per_rank * hidden;
  TORCH_CHECK(expert_output.size(1) == hidden, "ep: expert_output hidden mismatch");
  TORCH_CHECK(acc.numel() == chunk * world_size, "ep: acc.numel mismatch");
  TORCH_CHECK(
      scatter_idx.size(0) == num_tokens_per_rank * world_size,
      "ep: scatter_idx first dim must equal world_size * tokens");

  if (chunk == 0) {
    return output;
  }

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  const int32_t ws = static_cast<int32_t>(world_size);
  const int32_t r = static_cast<int32_t>(rank);
  const int32_t right = (r + 1) % ws;
  const int32_t topk = static_cast<int32_t>(scatter_idx.size(1));
  TORCH_CHECK(
      topk <= 32,
      "ep: topk must be <= 32 (owned-expert slots are tracked in a uint32 mask)");
  const uint32_t tag = static_cast<uint32_t>(iteration);
  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  constexpr int VEC_SIZE = 8;
  int64_t threads = hidden / VEC_SIZE;
  if (threads < 1) threads = 1;
  if (threads > 1024) threads = 1024;
  const int64_t eu_count = static_cast<int64_t>(
      queue.get_device().get_info<sycl::info::device::max_compute_units>());
  int32_t num_wg = 1;
  int64_t tokens_per_wg = num_tokens_per_rank;
  compute_launch_tokens(
      num_tokens_per_rank, hidden, threads, VEC_SIZE, eu_count, num_wg_hint,
      num_wg, tokens_per_wg);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ring_reduce_scatter_unpermute_ep", [&]() {
        auto kfn = RingReduceScatterUnpermuteEpKernel<scalar_t, VEC_SIZE>{
            expert_output.data_ptr<scalar_t>(),
            rank_buffers_ptr.data_ptr<int64_t>(),
            signal_pads_ptr.data_ptr<int64_t>(),
            acc.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            scatter_idx.data_ptr<int32_t>(),
            topk_idx.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            hidden,
            num_tokens_per_rank,
            tokens_per_wg,
            topk,
            r,
            ws,
            right,
            num_wg,
            tag,
            base_experts,
            rem_experts,
            boundary};
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
      "ring_reduce_scatter_unpermute(Tensor expert_output, Tensor rank_buffers_ptr, "
      "Tensor signal_pads_ptr, Tensor(a!) acc, Tensor(b!) output, "
      "Tensor scatter_idx, Tensor topk_weights, "
      "int rank, int world_size, int iteration, int num_wg_hint=-1) -> Tensor(b!)");
  m.def(
      "ring_reduce_scatter_unpermute_ep(Tensor expert_output, Tensor rank_buffers_ptr, "
      "Tensor signal_pads_ptr, Tensor(a!) acc, Tensor(b!) output, "
      "Tensor scatter_idx, Tensor topk_idx, Tensor topk_weights, "
      "int num_experts, int rank, int world_size, int iteration, int num_wg_hint=-1) -> Tensor(b!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_reduce_scatter_unpermute", ring_reduce_scatter_unpermute);
  m.impl("ring_reduce_scatter_unpermute_ep", ring_reduce_scatter_unpermute_ep);
}
