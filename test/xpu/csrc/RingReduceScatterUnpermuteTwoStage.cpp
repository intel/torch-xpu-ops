#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

// ===========================================================================
// Fused MoE unpermute + ring reduce-scatter  --  SINGLE-KERNEL, TWO-STAGE.
//
// This is a software-pipelined variant of RingReduceScatterUnpermute.cpp.
// It is still ONE kernel launch and uses the same push-based ring, signal
// pads and symmetric acc buffers; the only change is HOW the per-block work
// is scheduled so that the local unpermute compute overlaps the cross-rank
// wait instead of being serialized behind it.
//
// Why the original hides almost nothing (measured actual ~= comm + compute):
//   In the baseline every ring step does, per work-group:
//       wait(acc[b_t]) -> barrier -> unpermute(read expert_output + add acc +
//                                                store to the right peer)
//   The expensive weighted GATHER of expert_output (the read side, which is
//   ~256/288 MB of the compute traffic) is gated BEHIND the signal barrier, so
//   no work-group may start its gather for block b_t until that block's remote
//   partial has already arrived. The gather output IS the ring store, so the
//   compute and the transfer are strictly sequential and the critical path
//   collapses to (total_compute + total_comm).
//
// The overlap is realized by DOUBLE-BUFFERED CROSS-BLOCK PREFETCH. Two things
// that do NOT work (both measured): (a) simply reordering the gather ahead of
// the wait -- intra-work-group the gather loop retires before the lane-0 spin
// starts, and all work-groups are phase-synced so cross-WG TLP never fills the
// bubble (perf-neutral); (b) a producer/consumer subgroup split (subgroup 0
// waits while other subgroups gather) -- the waiter's per-iteration system-scope
// acquire fence plus the required global work-group barriers cost more than the
// overlap saves (slower). What DOES work: the ring stores are fire-and-forget,
// so after Stage C pushes block b_t into the peer we immediately issue the local
// gather for the NEXT block; that gather's HBM traffic overlaps the PCIe DRAIN
// of the b_t stores, i.e. the expensive compute is hidden behind our OUTGOING
// communication instead of racing the incoming wait.
//
// Schedule (per work-group), with `cur`/`nxt` a two-buffer scratch ping-pong:
//   Phase 0 (seed): fused unpermute(b0) -> right peer's acc; signal.
//   Prologue: gather(b_1) -> cur   (overlaps the seed store drain).
//   Steps t = 1..ws-1:
//     Stage B: wait for acc[b_t]  (cur already holds gather(b_t)).
//     Stage C: right.acc[b_t] = cur + acc[b_t]  (fire-and-forget remote stores);
//              then gather(b_{t+1}) -> nxt  (overlaps the Stage-C store drain);
//              release-fence + barrier + signal right; swap(cur, nxt).
//     Final step (t = ws-1): output = cur + acc[rank] (no push, no prefetch).
//
// Only the phase-0 seed gather (no communication to hide behind) stays exposed;
// every folded block's gather is prefetched a step ahead and hidden. `scratch`
// is a LOCAL [2, num_tokens_per_rank, hidden] ping-pong buffer.
//
// `scratch` is a LOCAL [2, num_tokens_per_rank, hidden] ping-pong buffer;
// work-groups write disjoint token ranges of each half, so the two buffers are
// safe to share across work-groups. It costs one extra local-HBM write (gather)
// + read (Stage C) per block (~2*tokens*hidden), which is small next to the
// topk*hidden gather read that it lets us hide behind the outgoing transfer.
//
// Topology / signalling are IDENTICAL to RingReduceScatterUnpermute.cpp:
//   right = (rank+1)%ws; slot(phase, wg) = phase * num_wg + wg; `iteration`
//   is a strictly increasing tag (pads zeroed + barriered by the Python
//   wrapper). Phase 0 seeds block b0=(rank-1+ws)%ws directly (no wait, nothing
//   to overlap); steps 1..ws-1 use the two-stage schedule above.
// ===========================================================================

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// See RingReduceScatterUnpermute.cpp for the full rationale: the peer-directed
// 16-byte store defaults to a plain (coherent) store; the writeback LSC path is
// opt-in via RING_LSC_STORE_WB and correctness-hazardous across the ring.
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

constexpr int32_t RING_MAX_WG = 64;

// 16-byte vector store with optional LSC cache control (see RING_LSC_STORE).
template <typename vec_t>
inline void ring_vec_store(vec_t* dst, vec_t vd) {
#if defined(RING_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  if constexpr (sizeof(vec_t) == 16) {
#if defined(RING_LSC_STORE_PLAIN)
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

inline void store_release_sys(uint32_t* addr, uint32_t val) {
  *addr = val;
  sycl::atomic_fence(
      sycl::memory_order::release, sycl::memory_scope::system);
}

inline void wait_eq_sys(uint32_t* addr, uint32_t val) {
  for (;;) {
    sycl::atomic_fence(
        sycl::memory_order::acquire, sycl::memory_scope::system);
    if (*addr == val)
      break;
  }
}

template <typename scalar_t, int VEC_SIZE>
struct RingReduceScatterUnpermuteTwoStageKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* expert_output_ptr;  // [num_tokens*topk, hidden] (local)
  const int64_t* rank_buffers_ptr;    // symm acc buffers, one per rank
  const int64_t* signal_pads_ptr;     // signal pads, one per rank
  scalar_t* acc_ptr;                  // == rank_buffers_ptr[rank]
  scalar_t* scratch_ptr;             // [2, num_tokens_per_rank, hidden] (local)
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

  // Stage A helper: weighted gather of one token from LOCAL expert_output into
  // `dst_row` (a local scratch row). No incoming partial is involved, and the
  // store is a plain local store (dst is not remote symmetric memory).
  inline void gather_token(
      int64_t gt, scalar_t* dst_row, int32_t lid, int32_t lsize) const {
    const int64_t hidden_vecs = hidden / VEC_SIZE;
    const int64_t topk_base = gt * topk;
    for (int64_t vh = lid; vh < hidden_vecs; vh += lsize) {
      const int64_t h_start = vh * VEC_SIZE;
      float acc[VEC_SIZE];
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i)
        acc[i] = 0.0f;
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        if (src_row < 0) continue;
        const float w = topk_weights_ptr[topk_base + k];
        const scalar_t* src =
            expert_output_ptr + static_cast<int64_t>(src_row) * hidden + h_start;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
          acc[i] += w * static_cast<float>(src[i]);
      }
      vec_t vd;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        const scalar_t s = static_cast<scalar_t>(acc[i]);
        vd[i] = *reinterpret_cast<const vec_elem_t*>(&s);
      }
      *reinterpret_cast<vec_t*>(dst_row + h_start) = vd;
    }
    for (int64_t h = hidden_vecs * VEC_SIZE + lid; h < hidden; h += lsize) {
      float a = 0.0f;
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

  // Baseline-style fused unpermute (read expert_output + optional add acc +
  // store) used only for the phase-0 seed, where there is nothing to overlap.
  inline void unpermute_token_seed(
      int64_t gt, scalar_t* dst_row, int32_t lid, int32_t lsize) const {
    const int64_t hidden_vecs = hidden / VEC_SIZE;
    const int64_t topk_base = gt * topk;
    for (int64_t vh = lid; vh < hidden_vecs; vh += lsize) {
      const int64_t h_start = vh * VEC_SIZE;
      float acc[VEC_SIZE];
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i)
        acc[i] = 0.0f;
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        if (src_row < 0) continue;
        const float w = topk_weights_ptr[topk_base + k];
        const scalar_t* src =
            expert_output_ptr + static_cast<int64_t>(src_row) * hidden + h_start;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
          acc[i] += w * static_cast<float>(src[i]);
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
      float a = 0.0f;
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

  // Stage C helper: dst_row = scratch_row + acc_row, elementwise. `scratch_row`
  // is local (plain load); `dst_row` is the right peer's acc (remote) or the
  // local output, so the store goes through ring_vec_store.
  inline void add_store_token(
      const scalar_t* scratch_row,
      const scalar_t* acc_row,
      scalar_t* dst_row,
      int32_t lid,
      int32_t lsize) const {
    const int64_t hidden_vecs = hidden / VEC_SIZE;
    for (int64_t vh = lid; vh < hidden_vecs; vh += lsize) {
      const int64_t h_start = vh * VEC_SIZE;
      vec_t vd;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        const float sum = static_cast<float>(scratch_row[h_start + i]) +
            static_cast<float>(acc_row[h_start + i]);
        const scalar_t s = static_cast<scalar_t>(sum);
        vd[i] = *reinterpret_cast<const vec_elem_t*>(&s);
      }
      ring_vec_store(reinterpret_cast<vec_t*>(dst_row + h_start), vd);
    }
    for (int64_t h = hidden_vecs * VEC_SIZE + lid; h < hidden; h += lsize) {
      const float sum =
          static_cast<float>(scratch_row[h]) + static_cast<float>(acc_row[h]);
      dst_row[h] = static_cast<scalar_t>(sum);
    }
  }

  // Stage A over this WG's token slice of `block`: gather -> scratch buffer
  // `scr` (one block wide, block-relative row offsets).
  inline void wg_gather_block(
      int32_t block,
      scalar_t* scr,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t gt =
          static_cast<int64_t>(block) * num_tokens_per_rank + local_t;
      scalar_t* dst_row = scr + local_t * hidden;
      gather_token(gt, dst_row, lid, lsize);
    }
  }

  // Stage C over this WG's token slice of `block`: right.acc[block] =
  // scr[block] + acc[block]. Row offset in acc/right_acc includes the block
  // dimension; the scratch buffer `scr` is block-relative (one block wide).
  inline void wg_add_store_block(
      int32_t block,
      const scalar_t* scr,
      scalar_t* dst_base,  // right peer's acc
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t acc_off =
          (static_cast<int64_t>(block) * num_tokens_per_rank + local_t) * hidden;
      const scalar_t* scratch_row = scr + local_t * hidden;
      const scalar_t* acc_row = acc_ptr + acc_off;
      scalar_t* dst_row = dst_base + acc_off;
      add_store_token(scratch_row, acc_row, dst_row, lid, lsize);
    }
  }

  // Stage C for the final block: output = scr[rank] + acc[rank]. Output is
  // compact ([tokens, hidden]) so its row offset is local-token relative.
  inline void wg_add_store_final(
      int32_t block,
      const scalar_t* scr,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t acc_off =
          (static_cast<int64_t>(block) * num_tokens_per_rank + local_t) * hidden;
      const scalar_t* scratch_row = scr + local_t * hidden;
      const scalar_t* acc_row = acc_ptr + acc_off;
      scalar_t* dst_row = output_ptr + local_t * hidden;
      add_store_token(scratch_row, acc_row, dst_row, lid, lsize);
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

    const int64_t chunk = num_tokens_per_rank * hidden;

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[rank]);
    uint32_t* right_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[right]);
    scalar_t* right_acc =
        reinterpret_cast<scalar_t*>(rank_buffers_ptr[right]);

    // Double-buffered scratch: `cur` holds the block being folded THIS step;
    // `nxt` receives the prefetched gather for the NEXT step.
    scalar_t* cur = scratch_ptr;
    scalar_t* nxt = scratch_ptr + chunk;

    // Phase 0: seed the partial for block b0 with our own contribution. There
    // is no incoming transfer yet, so nothing to overlap -- do the fused
    // unpermute directly into the right peer's acc with ALL lanes.
    {
      const int32_t b0 = (rank - 1 + world_size) % world_size;
      for (int64_t lt = 0; lt < token_cnt; ++lt) {
        const int64_t local_t = token_base + lt;
        const int64_t gt =
            static_cast<int64_t>(b0) * num_tokens_per_rank + local_t;
        const int64_t row_off =
            (static_cast<int64_t>(b0) * num_tokens_per_rank + local_t) * hidden;
        unpermute_token_seed(gt, right_acc + row_off, lid, lsize);
      }
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0) {
        store_release_sys(right_pad + (0 * num_wg + wg), tag);
      }
    }

    // Prologue: prefetch the gather for the first folded block (step t=1) into
    // `cur` while the seed's remote stores are still draining to the peer.
    if (world_size >= 2) {
      const int32_t b1 = (rank - 1 - 1 + 2 * world_size) % world_size;
      wg_gather_block(b1, cur, token_base, token_cnt, lid, lsize);
      // Publish `cur` (global) across subgroups before Stage C reads it.
      item.barrier(sycl::access::fence_space::global_and_local);
    }

    // Steps 1..ws-1: fold the incoming partial. The expensive local gather for
    // the NEXT block is issued right after this block's fire-and-forget remote
    // stores, so it overlaps the PCIe drain of those stores (compute hidden
    // behind our outgoing communication). Stage B (wait) and Stage C (add +
    // store) are the only things left on the ring critical path.
    for (int32_t t = 1; t < world_size; ++t) {
      const int32_t b_t = (rank - 1 - t + 2 * world_size) % world_size;

      // Stage B: wait for the incoming partial for b_t (`cur` already holds its
      // prefetched gather).
      if (lid == 0) {
        wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      if (t < world_size - 1) {
        // Stage C: right.acc[b_t] = cur + acc[b_t]. Remote stores are
        // fire-and-forget (no fence yet).
        wg_add_store_block(
            b_t, cur, right_acc, token_base, token_cnt, lid, lsize);

        // Publish the remote stores to the peer and signal it AS EARLY AS
        // POSSIBLE (before the prefetch) so the peer's wait is never delayed by
        // our local gather.
        sycl::atomic_fence(
            sycl::memory_order::release, sycl::memory_scope::system);
        item.barrier(sycl::access::fence_space::local_space);
        if (lid == 0) {
          store_release_sys(right_pad + (t * num_wg + wg), tag);
        }

        // Prefetch: gather the NEXT block into `nxt`. This local-HBM compute
        // now overlaps our idle time waiting for acc[b_{t+1}] to arrive.
        const int32_t b_next =
            (rank - 1 - (t + 1) + 2 * world_size) % world_size;
        wg_gather_block(b_next, nxt, token_base, token_cnt, lid, lsize);
        // Publish `nxt` (global) across subgroups before next step reads it.
        item.barrier(sycl::access::fence_space::global_and_local);

        scalar_t* tmp = cur;
        cur = nxt;
        nxt = tmp;
      } else {
        // Final: block `rank` (b_t == rank) is fully combined here.
        wg_add_store_final(b_t, cur, token_base, token_cnt, lid, lsize);
      }
    }
  }
};

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

// `acc` is rank's symmetric-memory buffer and must equal rank_buffers_ptr[rank].
// `scratch` is a LOCAL [2 * num_tokens_per_rank * hidden] ping-pong buffer used
// to prefetch each block's weighted gather one ring step ahead. `expert_output`
// and `output` are local (non-symmetric) tensors this rank owns.
at::Tensor ring_reduce_scatter_unpermute_two_stage(
    const at::Tensor& expert_output,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& signal_pads_ptr,
    at::Tensor acc,
    at::Tensor scratch,
    at::Tensor output,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    int64_t rank,
    int64_t world_size,
    int64_t iteration) {
  TORCH_CHECK(expert_output.dim() == 2, "two_stage: expert_output must be 2D [rows, hidden]");
  TORCH_CHECK(expert_output.is_contiguous(), "two_stage: expert_output must be contiguous");
  TORCH_CHECK(acc.dim() == 1, "two_stage: acc must be 1D");
  TORCH_CHECK(acc.is_contiguous(), "two_stage: acc must be contiguous");
  TORCH_CHECK(scratch.is_contiguous(), "two_stage: scratch must be contiguous");
  TORCH_CHECK(output.dim() == 2, "two_stage: output must be 2D [tokens, hidden]");
  TORCH_CHECK(output.is_contiguous(), "two_stage: output must be contiguous");
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "two_stage: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "two_stage: rank_buffers_ptr must be int64");
  TORCH_CHECK(
      signal_pads_ptr.dim() == 1 && signal_pads_ptr.size(0) == world_size,
      "two_stage: signal_pads_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      signal_pads_ptr.scalar_type() == at::kLong,
      "two_stage: signal_pads_ptr must be int64");
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "two_stage: scatter_idx must be 2D [num_tokens, topk]");
  TORCH_CHECK(scatter_idx.scalar_type() == at::kInt, "two_stage: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(
      topk_weights.dim() == 2 && topk_weights.sizes() == scatter_idx.sizes(),
      "two_stage: topk_weights must match scatter_idx shape");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "two_stage: topk_weights must be float32");
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(rank >= 0 && rank < world_size, "two_stage: rank out of range");
  TORCH_CHECK(
      expert_output.scalar_type() == output.scalar_type() &&
          expert_output.scalar_type() == acc.scalar_type() &&
          expert_output.scalar_type() == scratch.scalar_type(),
      "two_stage: expert_output/acc/scratch/output must have same dtype");
  TORCH_CHECK(iteration > 0, "two_stage: iteration must be > 0");

  const int64_t num_tokens_per_rank = output.size(0);
  const int64_t hidden = output.size(1);
  const int64_t chunk = num_tokens_per_rank * hidden;

  TORCH_CHECK(
      expert_output.size(1) == hidden,
      "two_stage: expert_output hidden must match output");
  TORCH_CHECK(
      acc.numel() == chunk * world_size,
      "two_stage: acc.numel() must equal tokens*hidden*world_size");
  TORCH_CHECK(
      scratch.numel() == 2 * chunk,
      "two_stage: scratch.numel() must equal 2*num_tokens_per_rank*hidden");
  TORCH_CHECK(
      scatter_idx.size(0) == num_tokens_per_rank * world_size,
      "two_stage: scatter_idx first dim must equal world_size * tokens");

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
  constexpr int64_t threads = 256;
  int32_t num_wg = 1;
  int64_t tokens_per_wg = num_tokens_per_rank;
  compute_launch_tokens(
      num_tokens_per_rank, hidden, threads, VEC_SIZE, num_wg, tokens_per_wg);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ring_reduce_scatter_unpermute_two_stage", [&]() {
        auto kfn =
            RingReduceScatterUnpermuteTwoStageKernel<scalar_t, VEC_SIZE>{
                expert_output.data_ptr<scalar_t>(),
                rank_buffers_ptr.data_ptr<int64_t>(),
                signal_pads_ptr.data_ptr<int64_t>(),
                acc.data_ptr<scalar_t>(),
                scratch.data_ptr<scalar_t>(),
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

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ring_reduce_scatter_unpermute_two_stage(Tensor expert_output, "
      "Tensor rank_buffers_ptr, Tensor signal_pads_ptr, Tensor(a!) acc, "
      "Tensor(c!) scratch, Tensor(b!) output, Tensor scatter_idx, "
      "Tensor topk_weights, int rank, int world_size, int iteration) "
      "-> Tensor(b!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl(
      "ring_reduce_scatter_unpermute_two_stage",
      ring_reduce_scatter_unpermute_two_stage);
}
