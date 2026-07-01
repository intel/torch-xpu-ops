#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <cstdlib>
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

// allgather + permute moves hidden vectors by pure byte copies (no arithmetic),
// so it is dtype-agnostic and additionally supports FP16 and the two 1-byte FP8
// formats (e4m3fn / e5m2).  Used by the ring permute kernels below.
#ifndef AT_DISPATCH_PERMUTE_DTYPES
#define AT_DISPATCH_PERMUTE_DTYPES(scalar_type, name, ...)            \
  AT_DISPATCH_SWITCH(                                                 \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kHalf, __VA_ARGS__);                       \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__);                      \
      AT_DISPATCH_CASE(at::kFloat8_e4m3fn, __VA_ARGS__);              \
      AT_DISPATCH_CASE(at::kFloat8_e5m2, __VA_ARGS__))
#endif

namespace {

constexpr int32_t RING_MAX_WG = 1024;

// The cross-rank traffic in the PULL ring is a remote READ from the left peer's
// gather slot followed by a LOCAL write to our gather slot. That local write is
// NOT purely local: on the next hop the RIGHT peer remote-reads the very block
// we just wrote, so the write must reach a system coherence point before we
// signal. An LSC store with L1WB_L3WB (writeback) cache control leaves the data
// in the writeback cache; the system-scope `release` fence orders it but does
// NOT reliably drain it to where a remote peer's load observes it in time. The
// result is a multi-hop correctness bug: hop 0/1 (data published locally by the
// owner) are fine, but blocks relayed two or more hops are read stale by the
// downstream peer. The write therefore must be a coherent (non-writeback) store
// by default.
// Measured on a PCIe fabric (8x B60, bf16):
//   - LSC store (L1WB_L3WB) on the LOCAL write:   ~3.43 -> ~3.31 ms (~3-4% win)
//   - LSC L1UC load on the REMOTE read:           ~3.43 -> ~3.89 ms (~13% LOSS)
// The cached/pipelined remote read beats an L1UC bypass, so the remote load
// stays a plain load. The local store defaults to a plain (coherent) store
// because its data is consumed by a remote peer; the ~3-4% writeback win is not
// worth the correctness hazard.
// Opt-ins (only safe on fabrics where the writeback drains before the peer
// read): RING_LSC_COPY_LSC_STORE (writeback store), RING_LSC_COPY_LSC_LOAD
// (LSC load). Opt-out: RING_NO_LSC_COPY (disable LSC builtins entirely).
#if !defined(RING_LSC_COPY) && !defined(RING_NO_LSC_COPY)
#define RING_LSC_COPY 1
#endif
#if defined(RING_LSC_COPY) && !defined(RING_LSC_COPY_LSC_LOAD)
#define RING_LSC_COPY_PLAIN_LOAD 1
#endif
#if defined(RING_LSC_COPY) && !defined(RING_LSC_COPY_LSC_STORE)
#define RING_LSC_COPY_PLAIN_STORE 1
#endif

#if defined(RING_LSC_COPY) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
typedef uint32_t ring_lsc_u4 __attribute__((ext_vector_type(4)));
enum RingLscLdcc { RING_LSC_LDCC_L1UC_L3C = 2 };
enum RingLscStcc { RING_LSC_STCC_L1WB_L3WB = 7 };
SYCL_EXTERNAL extern "C" ring_lsc_u4 __builtin_IB_lsc_load_global_uint4(
    const __attribute__((opencl_global)) ring_lsc_u4* base,
    int off,
    enum RingLscLdcc cc);
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_uint4(
    __attribute__((opencl_global)) ring_lsc_u4* base,
    int off,
    ring_lsc_u4 val,
    enum RingLscStcc cc);
#endif

// 16-byte copy (load remote + store local) with optional LSC cache control.
template <typename vec_t>
inline void ring_vec_copy(vec_t* dst, const vec_t* src) {
#if defined(RING_LSC_COPY) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  if constexpr (sizeof(vec_t) == 16) {
#if defined(RING_LSC_COPY_PLAIN_LOAD)
    ring_lsc_u4 v = *reinterpret_cast<const ring_lsc_u4*>(src);
#else
    ring_lsc_u4 v = __builtin_IB_lsc_load_global_uint4(
        (const __attribute__((opencl_global)) ring_lsc_u4*)(src),
        0,
        RING_LSC_LDCC_L1UC_L3C);
#endif
#if defined(RING_LSC_COPY_PLAIN_STORE)
    *reinterpret_cast<ring_lsc_u4*>(dst) = v;
#else
    __builtin_IB_lsc_store_global_uint4(
        (__attribute__((opencl_global)) ring_lsc_u4*)(dst),
        0,
        v,
        RING_LSC_STCC_L1WB_L3WB);
#endif
  } else {
    *dst = *src;
  }
#else
  *dst = *src;
#endif
}

// 16-byte store with optional LSC cache control (L1WB_L3WB). Used for the PUSH
// kernel's posted remote write: contiguous remote writes get combined through
// L3 into larger burst transactions over the cross-GPU link (same lever as the
// reduce-scatter push). The value is already in a register (reused for the
// local permute writes), so this is store-only.
template <typename vec_t>
inline void ring_vec_store_remote(vec_t* dst, vec_t vd) {
#if defined(RING_LSC_COPY) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  if constexpr (sizeof(vec_t) == 16) {
    ring_lsc_u4 v = *reinterpret_cast<ring_lsc_u4*>(&vd);
    __builtin_IB_lsc_store_global_uint4(
        (__attribute__((opencl_global)) ring_lsc_u4*)(dst),
        0,
        v,
        RING_LSC_STCC_L1WB_L3WB);
  } else {
    *dst = vd;
  }
#else
  *dst = vd;
#endif
}
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
  using vec_elem_t = std::conditional_t<
      sizeof(scalar_t) == 1,
      uint8_t,
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_shard_ptr;   // [num_tokens_per_rank, hidden]
  const int64_t* rank_buffers_ptr;   // symm gather buffers, one per rank
  const int64_t* signal_pads_ptr;    // signal pads, one per rank
  scalar_t* gather_ptr;              // == rank_buffers_ptr[rank]
  scalar_t* remap_ptr;              // [remap_rows, hidden] (local)
  const int32_t* scatter_idx_ptr;  // [world_size * num_tokens_per_rank, topk] (absolute)
  int64_t hidden;
  int64_t num_tokens_per_rank;
  int64_t tokens_per_wg;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t left;
  int32_t num_wg;
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
        ring_vec_copy(&dv[i], &sv[i]);
      for (int64_t i = nv * VEC_SIZE + lid; i < n; i += lsize)
        dst[i] = src[i];
    } else {
      for (int64_t i = lid; i < n; i += lsize)
        dst[i] = src[i];
    }
  }

  // Permute every token in this work-group's slice of gathered block `block`
  // into the expert-sorted `remap` buffer.  For each top-k slot, the token's
  // hidden vector is copied to remap[scatter_idx[gt * topk + k]].
  // scatter_idx holds absolute destination rows (same as AllgatherPermuteRingVecKernel).
  //
  // Each lane loads its slice of the source hidden vector ONCE into a register
  // and writes it to all top-k destinations, instead of re-reading the source
  // from the gather buffer once per top-k slot (topk reads -> 1 read).  This
  // matches the read-once/write-topk pattern of the standalone allgather_permute
  // kernel and keeps the gather-buffer read traffic at num_tokens*hidden.
  inline void wg_permute_block(
      int32_t block,
      int64_t token_base,
      int64_t token_cnt,
      int32_t lid,
      int32_t lsize) const {
    const uintptr_t base_align =
        reinterpret_cast<uintptr_t>(gather_ptr) |
        reinterpret_cast<uintptr_t>(remap_ptr);
    const bool vectorizable =
        ((base_align % (VEC_SIZE * sizeof(scalar_t))) == 0) &&
        (hidden % VEC_SIZE == 0);
    const int64_t hidden_vecs = hidden / VEC_SIZE;

    for (int64_t lt = 0; lt < token_cnt; ++lt) {
      const int64_t local_t = token_base + lt;
      const int64_t gt =
          static_cast<int64_t>(block) * num_tokens_per_rank + local_t;
      const scalar_t* src = gather_ptr + gt * hidden;
      const int64_t topk_base = gt * topk;

      if (vectorizable) {
        auto sv = reinterpret_cast<const vec_t*>(src);
        for (int64_t i = lid; i < hidden_vecs; i += lsize) {
          const vec_t v = sv[i];  // single load, reused for every top-k slot
          for (int32_t k = 0; k < topk; ++k) {
            const int32_t dst_row = scatter_idx_ptr[topk_base + k];
            if (dst_row < 0) continue;
            auto dv = reinterpret_cast<vec_t*>(
                remap_ptr + static_cast<int64_t>(dst_row) * hidden);
            dv[i] = v;
          }
        }
      } else {
        for (int64_t i = lid; i < hidden; i += lsize) {
          const scalar_t v = src[i];  // single load, reused for every top-k slot
          for (int32_t k = 0; k < topk; ++k) {
            const int32_t dst_row = scatter_idx_ptr[topk_base + k];
            if (dst_row < 0) continue;
            remap_ptr[static_cast<int64_t>(dst_row) * hidden + i] = v;
          }
        }
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
    scalar_t* left_gather =
        reinterpret_cast<scalar_t*>(rank_buffers_ptr[left]);

    // Phase 0: publish our own shard (block `rank`) into our LOCAL gather slot,
    // then tell the right peer it may pull block `rank` from us.  Permute it
    // while the peers read.  (PULL ring: data moves by REMOTE READS from the
    // left neighbour rather than remote writes to the right neighbour, because
    // on this PCIe fabric peer reads sustain far higher bandwidth than the
    // posted-write traffic a push ring generates.)
    {
      const scalar_t* src = input_shard_ptr + base;
      const int64_t slot = static_cast<int64_t>(rank) * chunk + base;
      wg_copy(src, gather_ptr + slot, cnt, lid, lsize);
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (lid == 0 && world_size > 1) {
        store_release_sys(right_pad + (0 * num_wg + wg), tag);
      }
      // Terminal local permute of block `rank` (overlaps peers' reads).
      wg_permute_block(rank, token_base, token_cnt, lid, lsize);
    }

    // Steps 1..ws-1: wait until the left peer has block `idx` ready, PULL it
    // into our gather slot, re-publish to the right peer, then permute it.
    for (int32_t t = 1; t < world_size; ++t) {
      if (lid == 0) {
        wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
      }
      item.barrier(sycl::access::fence_space::local_space);
      sycl::atomic_fence(
          sycl::memory_order::acquire, sycl::memory_scope::system);

      const int32_t idx = (rank - t + world_size) % world_size;

      // Remote read: copy block `idx` from the left peer's gather slot into our
      // own.  This is the only cross-rank traffic and it is a pure load.
      const scalar_t* src =
          left_gather + static_cast<int64_t>(idx) * chunk + base;
      scalar_t* dst = gather_ptr + static_cast<int64_t>(idx) * chunk + base;
      wg_copy(src, dst, cnt, lid, lsize);
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
      item.barrier(sycl::access::fence_space::local_space);
      if (t < world_size - 1 && lid == 0) {
        store_release_sys(right_pad + (t * num_wg + wg), tag);
      }
      // Permute this block's tokens (terminal local work for all blocks).
      wg_permute_block(idx, token_base, token_cnt, lid, lsize);
    }
  }
};

// ===========================================================================
// PUSH variant: forward each block to the RIGHT peer with posted remote writes
// that are INTERLEAVED with the permute.  For every gathered vec we issue one
// posted write to the right peer's gather slot (drains asynchronously over the
// fabric) and then the topk local remap writes -- so the write drain overlaps
// the HBM permute on the SAME work-items (no separate comm phase, no gather
// readback).  Compare against the PULL kernel above on a given fabric: PULL
// wins when peer reads >> posted writes (the PCIe case the default assumes),
// PUSH wins when posted writes are cheap and overlap matters.
// ===========================================================================
template <typename scalar_t, int VEC_SIZE>
struct RingAllgatherPermutePushKernel {
  using vec_elem_t = std::conditional_t<
      sizeof(scalar_t) == 1,
      uint8_t,
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_shard_ptr;   // [num_tokens_per_rank, hidden] own shard
  const int64_t* rank_buffers_ptr;   // symm gather buffers, one per rank
  const int64_t* signal_pads_ptr;    // signal pads, one per rank
  scalar_t* gather_ptr;              // == rank_buffers_ptr[rank] (left pushes here)
  scalar_t* remap_ptr;              // [remap_rows, hidden] (local)
  const int32_t* scatter_idx_ptr;  // absolute destination rows
  int64_t hidden;
  int64_t num_tokens_per_rank;
  int64_t tokens_per_wg;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t right;
  int32_t num_wg;
  uint32_t tag;
  bool lsc_store;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    const int64_t token_base = static_cast<int64_t>(wg) * tokens_per_wg;
    int64_t token_cnt = num_tokens_per_rank - token_base;
    if (token_cnt > tokens_per_wg) token_cnt = tokens_per_wg;
    if (token_cnt < 0) token_cnt = 0;

    const int64_t chunk = num_tokens_per_rank * hidden;
    const int64_t hidden_vecs = hidden / VEC_SIZE;
    const bool vectorizable =
        ((reinterpret_cast<uintptr_t>(gather_ptr) |
          reinterpret_cast<uintptr_t>(remap_ptr) |
          reinterpret_cast<uintptr_t>(input_shard_ptr) |
          static_cast<uintptr_t>(rank_buffers_ptr[right])) %
             (VEC_SIZE * sizeof(scalar_t)) ==
         0) &&
        (hidden % VEC_SIZE == 0);

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[rank]);
    uint32_t* right_pad = reinterpret_cast<uint32_t*>(signal_pads_ptr[right]);
    scalar_t* right_gather =
        reinterpret_cast<scalar_t*>(rank_buffers_ptr[right]);

    // Each rank pushes its own block first, then relays blocks it receives from
    // the left.  At step t it owns block (rank - t); for t>0 it must wait until
    // the left peer has pushed that block into this rank's gather slot.
    for (int32_t t = 0; t < world_size; ++t) {
      const int32_t block = (rank - t + world_size) % world_size;
      const scalar_t* src;
      if (t == 0) {
        src = input_shard_ptr;  // own shard
      } else {
        if (lid == 0) {
          wait_eq_sys(my_pad + ((t - 1) * num_wg + wg), tag);
        }
        item.barrier(sycl::access::fence_space::local_space);
        sycl::atomic_fence(
            sycl::memory_order::acquire, sycl::memory_scope::system);
        src = gather_ptr + static_cast<int64_t>(block) * chunk;
      }

      const bool fwd = (t < world_size - 1);
      scalar_t* fwd_base = right_gather + static_cast<int64_t>(block) * chunk;

      for (int64_t lt = 0; lt < token_cnt; ++lt) {
        const int64_t local_t = token_base + lt;
        const scalar_t* s = src + local_t * hidden;
        scalar_t* d = fwd_base + local_t * hidden;
        const int64_t gt =
            static_cast<int64_t>(block) * num_tokens_per_rank + local_t;
        const int64_t topk_base = gt * topk;
        if (vectorizable) {
          auto sv = reinterpret_cast<const vec_t*>(s);
          auto dv = reinterpret_cast<vec_t*>(d);
          for (int64_t i = lid; i < hidden_vecs; i += lsize) {
            const vec_t v = sv[i];        // single load
            if (fwd) {
              // LSC write-back store on the remote forward write only pays off
              // when comm volume is high (many ring hops).  At low world_size
              // the kernel is dominated by the local topk scatter writes below,
              // and write-back-caching the remote line pollutes L1/L3 and steals
              // bandwidth from that dominant stream (measured ~+18% at ws=4).
              // So use the LSC store only when host enabled it (ws gated).
              if (lsc_store) {
                ring_vec_store_remote(&dv[i], v);  // posted remote write (async drain)
              } else {
                dv[i] = v;                 // plain remote write
              }
            }
            for (int32_t k = 0; k < topk; ++k) {
              const int32_t dst_row = scatter_idx_ptr[topk_base + k];
              if (dst_row < 0) continue;
              auto rv = reinterpret_cast<vec_t*>(
                  remap_ptr + static_cast<int64_t>(dst_row) * hidden);
              rv[i] = v;                  // local HBM write (overlaps drain)
            }
          }
        } else {
          for (int64_t i = lid; i < hidden; i += lsize) {
            const scalar_t v = s[i];
            if (fwd) d[i] = v;
            for (int32_t k = 0; k < topk; ++k) {
              const int32_t dst_row = scatter_idx_ptr[topk_base + k];
              if (dst_row < 0) continue;
              remap_ptr[static_cast<int64_t>(dst_row) * hidden + i] = v;
            }
          }
        }
      }

      if (fwd) {
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
// local (non-symmetric) expert-sorted buffer this rank owns.
at::Tensor ring_allgather_permute(
    const at::Tensor& input_shard,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& signal_pads_ptr,
    at::Tensor gather_output,
    at::Tensor remap_output,
    const at::Tensor& scatter_idx,
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
  TORCH_CHECK(
      scatter_idx.dim() == 2,
      "ring_allgather_permute: scatter_idx must be 2D [num_tokens, topk]");
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
      scatter_idx.size(0) == num_tokens_per_rank * world_size,
      "ring_allgather_permute: scatter_idx first dim must equal world_size * tokens");

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
  const int32_t left = (r - 1 + ws) % ws;
  const int32_t topk = static_cast<int32_t>(scatter_idx.size(1));
  const uint32_t tag = static_cast<uint32_t>(iteration);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;
  int32_t num_wg = 1;
  int64_t tokens_per_wg = num_tokens_per_rank;
  compute_launch_tokens(
      num_tokens_per_rank, hidden, threads, VEC_SIZE, num_wg, tokens_per_wg);

  AT_DISPATCH_PERMUTE_DTYPES(
      remap_output.scalar_type(), "ring_allgather_permute", [&]() {
        auto kfn = RingAllgatherPermuteSingleKernel<scalar_t, VEC_SIZE>{
            input_shard.data_ptr<scalar_t>(),
            rank_buffers_ptr.data_ptr<int64_t>(),
            signal_pads_ptr.data_ptr<int64_t>(),
            gather_output.data_ptr<scalar_t>(),
            remap_output.data_ptr<scalar_t>(),
            scatter_idx.data_ptr<int32_t>(),
            hidden,
            num_tokens_per_rank,
            tokens_per_wg,
            topk,
            r,
            ws,
            right,
            left,
            num_wg,
            tag};
        sycl_kernel_submit(
            sycl::range<1>(static_cast<size_t>(num_wg) * threads),
            sycl::range<1>(threads),
            queue,
            kfn);
      });

  return remap_output;
}

// PUSH variant of ring_allgather_permute (identical signature/semantics).
at::Tensor ring_allgather_permute_push(
    const at::Tensor& input_shard,
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& signal_pads_ptr,
    at::Tensor gather_output,
    at::Tensor remap_output,
    const at::Tensor& scatter_idx,
    int64_t rank,
    int64_t world_size,
    int64_t iteration) {
  TORCH_CHECK(input_shard.dim() == 2 && input_shard.is_contiguous());
  TORCH_CHECK(gather_output.dim() == 1 && gather_output.is_contiguous());
  TORCH_CHECK(remap_output.dim() == 2 && remap_output.is_contiguous());
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size &&
      rank_buffers_ptr.scalar_type() == at::kLong);
  TORCH_CHECK(
      signal_pads_ptr.dim() == 1 && signal_pads_ptr.size(0) == world_size &&
      signal_pads_ptr.scalar_type() == at::kLong);
  TORCH_CHECK(scatter_idx.dim() == 2 && scatter_idx.scalar_type() == at::kInt &&
              scatter_idx.is_contiguous());
  TORCH_CHECK(rank >= 0 && rank < world_size);
  TORCH_CHECK(iteration > 0);

  const int64_t num_tokens_per_rank = input_shard.size(0);
  const int64_t hidden = input_shard.size(1);
  const int64_t chunk = num_tokens_per_rank * hidden;
  TORCH_CHECK(gather_output.numel() == chunk * world_size);
  TORCH_CHECK(remap_output.size(1) == hidden);
  TORCH_CHECK(scatter_idx.size(0) == num_tokens_per_rank * world_size);
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
  const int32_t topk = static_cast<int32_t>(scatter_idx.size(1));
  const uint32_t tag = static_cast<uint32_t>(iteration);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;
  int32_t num_wg = 1;
  int64_t tokens_per_wg = num_tokens_per_rank;
  compute_launch_tokens(
      num_tokens_per_rank, hidden, threads, VEC_SIZE, num_wg, tokens_per_wg);

  AT_DISPATCH_PERMUTE_DTYPES(
      remap_output.scalar_type(), "ring_allgather_permute_push", [&]() {
        // LSC L1WB_L3WB store on the remote forward write helps only when comm
        // volume is high (many ring hops).  At low world_size the push kernel
        // is dominated by the local topk scatter and the write-back-cached
        // remote line hurts (measured ~+18% at ws=4), so gate on world_size.
        // Override with RING_AGP_LSC=0/1.
        bool lsc_store = ws > 4;
        if (const char* e = std::getenv("RING_AGP_LSC")) {
          lsc_store = (e[0] != '0');
        }
        auto kfn = RingAllgatherPermutePushKernel<scalar_t, VEC_SIZE>{
            input_shard.data_ptr<scalar_t>(),
            rank_buffers_ptr.data_ptr<int64_t>(),
            signal_pads_ptr.data_ptr<int64_t>(),
            gather_output.data_ptr<scalar_t>(),
            remap_output.data_ptr<scalar_t>(),
            scatter_idx.data_ptr<int32_t>(),
            hidden,
            num_tokens_per_rank,
            tokens_per_wg,
            topk,
            r,
            ws,
            right,
            num_wg,
            tag,
            lsc_store};
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
      "Tensor scatter_idx, int rank, int world_size, int iteration) -> Tensor(b!)");
  m.def(
      "ring_allgather_permute_push(Tensor input_shard, Tensor rank_buffers_ptr, "
      "Tensor signal_pads_ptr, Tensor(a!) gather_output, Tensor(b!) remap_output, "
      "Tensor scatter_idx, int rank, int world_size, int iteration) -> Tensor(b!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ring_allgather_permute", ring_allgather_permute);
  m.impl("ring_allgather_permute_push", ring_allgather_permute_push);
}
