#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>
#include <cstdlib>

// ===========================================================================
// MoE owner-sparse COMBINE WITHOUT unpermute -- SINGLE fused push+reduce kernel,
// built strictly on the RingReduceScatter.cpp machinery (push-based transfer,
// per-work-group signal pads with system-scope release/acquire, L3-combined LSC
// stores) but SPARSE, i.e. the exact inverse of ep_dispatch_no_permute.
//
// Dispatch pushed each source token (s, t) ONLY to the distinct owner ranks of
// its top-k experts.  Combine reverses that exact path: every owner rank pushes
// its (owner-weighted) contribution for the rows it OWNS back to the source
// rank -- so the cross-device volume scales with owned data (~76MB) instead of
// the dense (W-1)*T*H of a reduce-scatter.
//
// A token can be owned by several ranks, so their contributions must be summed.
// Like RingReduceScatter the reduction is FUSED into the single kernel (no
// second pass): each owner writes into its OWN private per-token slot on the
// destination, contiguous per token, and the destination sums the W slots:
//
//     recv[t*W + owner]  <-- owner's contribution for the source's local token t
//     output[t] = sum_{o=0..W-1} recv[t*W + o]
//
// Work-group `g` owns a token slice [t0, t1) of the LOCAL T tokens and, exactly
// like RingReduceScatter, only ever interacts with the SAME work-group index on
// its peers (no cross-wg dependency => no deadlock):
//   Phase 1 (push, as OWNER): for each source rank s, push the rows this rank
//     owns among s's slice-g tokens into s's recv slot; then signal s's pad at
//     slot (g*W + rank).  Non-owned slots on the destination stay zero.
//   Phase 2 (reduce, as SOURCE): wait until all W owners have signaled slice g
//     (system-scope acquire), then sum the DISTINCT owner-slots of every
//     slice-g token into output (only slots an owner actually wrote are read).
// Because signals are per-slice, a work-group can reduce its slice as soon as
// its owners are done while other work-groups are still pushing -- so the reduce
// overlaps the push and the whole combine tracks the push cost (< the dense
// reduce-scatter it replaces), instead of push + a serial reduce pass.
//
// A single leading host barrier per call (negligible, ~0.01ms) fences the
// previous call's reduce reads against this call's remote pushes; the strictly
// increasing `iteration` tag means the signal pads never need resetting.
// ===========================================================================

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// L3-combined 16-byte remote store (see RingReduceScatter RING_LSC_STORE).
#if !defined(EPC_NP_LSC_STORE) && !defined(EPC_NP_NO_LSC_STORE)
#define EPC_NP_LSC_STORE 1
#endif

#if defined(EPC_NP_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
typedef uint32_t epc_np_lsc_u4 __attribute__((ext_vector_type(4)));
enum EpcNpLscStcc { EPC_NP_LSC_STCC_L1WB_L3WB = 7 };
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_uint4(
    __attribute__((opencl_global)) epc_np_lsc_u4* base,
    int off,
    epc_np_lsc_u4 val,
    enum EpcNpLscStcc cc);
#endif

namespace {

template <typename vec_t>
inline void epc_np_vec_store(vec_t* dst, vec_t vd) {
#if defined(EPC_NP_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  if constexpr (sizeof(vec_t) == 16) {
    epc_np_lsc_u4 v = *reinterpret_cast<epc_np_lsc_u4*>(&vd);
    __builtin_IB_lsc_store_global_uint4(
        (__attribute__((opencl_global)) epc_np_lsc_u4*)(dst),
        0,
        v,
        EPC_NP_LSC_STCC_L1WB_L3WB);
  } else {
    *dst = vd;
  }
#else
  *dst = vd;
#endif
}

// Upper bound on work-groups; the Python wrapper sizes the signal pad as
// world_size * this, so num_wg * world_size (our pad usage) must fit.
constexpr int32_t EPC_NP_MAX_WG = 256;

// Upper bound on top-k, used to size the small per-token owner list computed on
// the reduce path (kept in registers, no SLM).
constexpr int32_t EPC_NP_MAX_TOPK = 32;

// Signal helpers -- identical semantics to RingReduceScatter / xccl Signal.hpp:
// store the value THEN a system release fence; acquire fence BEFORE each load.
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

// SINGLE-KERNEL owner-sparse combine (PUSH based + fused reduce).
template <typename scalar_t, int VEC_SIZE>
struct EpCombineNoPermuteKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* contributions;  // [W*T, H] this rank's owner-weighted rows
  const int32_t* topk_idx_full;   // [W*T, topk] routing for every token
  const int64_t* recv_buffers;    // [W] each rank's [T*W, H] recv buffer
  const int64_t* signal_pads;     // [W] each rank's uint32 pad region
  const scalar_t* recv_local;     // recv_buffers[rank]; peers push here
  scalar_t* output;               // [T, H]
  int32_t tokens_per_rank;        // T
  int32_t hidden;
  int32_t hidden_vecs;
  int32_t topk;
  int32_t tokens_per_wg;
  int32_t rank;
  int32_t world_size;
  int32_t num_wg;
  uint32_t tag;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;
  // This rank's owned experts form a CONTIGUOUS range [expert_lo, expert_hi).
  // The push ownership test (executed redundantly by every work-item for every
  // source token) then reduces to two comparisons instead of the two integer
  // divisions inside owner_of -- a large ALU saving on the hot path.
  int32_t expert_lo;
  int32_t expert_hi;

  inline int32_t owner_of(int32_t expert) const {
    if (expert < boundary) {
      return expert / (base_experts + 1);
    }
    return rem_experts + (expert - boundary) / base_experts;
  }

  void operator()(sycl::nd_item<1> item) const {
    const int32_t wg = static_cast<int32_t>(item.get_group(0));
    const int32_t lid = static_cast<int32_t>(item.get_local_id(0));
    const int32_t lsize = static_cast<int32_t>(item.get_local_range(0));

    const int32_t t0 = wg * tokens_per_wg;
    int32_t t1 = t0 + tokens_per_wg;
    if (t1 > tokens_per_rank) {
      t1 = tokens_per_rank;
    }

    uint32_t* my_pad = reinterpret_cast<uint32_t*>(signal_pads[rank]);

    // ---- Phase 1: push owned rows to every source, then signal it. -------- //
    // Stagger the destination order by work-group so that at any instant the
    // work-groups are spread across the W cross-device links (all starting at
    // dest 0 would serialize every push onto a single link).
    for (int32_t si = 0; si < world_size; ++si) {
      const int32_t s = (wg + si) % world_size;
      scalar_t* sbuf = reinterpret_cast<scalar_t*>(recv_buffers[s]);
      const bool remote = (s != rank);
      for (int32_t t = t0; t < t1; ++t) {
        const int32_t row = s * tokens_per_rank + t;  // source (s, t)
        bool owned = false;
        for (int32_t k = 0; k < topk; ++k) {
          const int32_t e = topk_idx_full[row * topk + k];
          if (e >= expert_lo && e < expert_hi) {
            owned = true;
            break;
          }
        }
        if (!owned) {
          continue;
        }
        const scalar_t* src =
            contributions + static_cast<int64_t>(row) * hidden;
        // This owner's private slot on source s: row (t*W + rank).
        scalar_t* dst = sbuf +
            (static_cast<int64_t>(t) * world_size + rank) * hidden;
        const vec_t* svec = reinterpret_cast<const vec_t*>(src);
        vec_t* dvec = reinterpret_cast<vec_t*>(dst);
        if (remote) {
          for (int32_t i = lid; i < hidden_vecs; i += lsize) {
            epc_np_vec_store(&dvec[i], svec[i]);
          }
        } else {
          for (int32_t i = lid; i < hidden_vecs; i += lsize) {
            dvec[i] = svec[i];
          }
        }
      }
    }
    // One system release fence makes all of this work-group's slice-g pushes
    // visible, then signal every source's pad (slot g*W + rank).
    sycl::atomic_fence(
        sycl::memory_order::release, sycl::memory_scope::system);
    item.barrier(sycl::access::fence_space::local_space);
    if (lid == 0) {
      for (int32_t s = 0; s < world_size; ++s) {
        uint32_t* s_pad = reinterpret_cast<uint32_t*>(signal_pads[s]);
        s_pad[wg * world_size + rank] = tag;
      }
      sycl::atomic_fence(
          sycl::memory_order::release, sycl::memory_scope::system);
    }

    // ---- Phase 2: once every owner signaled slice g, reduce our tokens. --- //
    if (lid == 0) {
      for (int32_t o = 0; o < world_size; ++o) {
        wait_eq_sys(my_pad + (wg * world_size + o), tag);
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
    sycl::atomic_fence(
        sycl::memory_order::acquire, sycl::memory_scope::system);

    for (int32_t t = t0; t < t1; ++t) {
      // Only this token's DISTINCT owners ever pushed a slot; summing just those
      // (instead of all W slots) skips the never-written slots -- less local
      // read bandwidth AND correct without relying on the non-owned slots being
      // zero (so the recv buffer need not be re-zeroed under dynamic routing).
      const int32_t lrow = rank * tokens_per_rank + t;  // this rank's token t
      int32_t owners[EPC_NP_MAX_TOPK];
      int32_t num_owners = 0;
      for (int32_t k = 0; k < topk; ++k) {
        const int32_t o = owner_of(topk_idx_full[lrow * topk + k]);
        bool dup = false;
        for (int32_t j = 0; j < num_owners; ++j) {
          if (owners[j] == o) {
            dup = true;
            break;
          }
        }
        if (!dup) {
          owners[num_owners++] = o;
        }
      }
      const scalar_t* base =
          recv_local + static_cast<int64_t>(t) * world_size * hidden;
      scalar_t* out = output + static_cast<int64_t>(t) * hidden;
      vec_t* ovec = reinterpret_cast<vec_t*>(out);
      for (int32_t i = lid; i < hidden_vecs; i += lsize) {
        sycl::vec<float, VEC_SIZE> acc(0.0f);
        for (int32_t oi = 0; oi < num_owners; ++oi) {
          const int32_t o = owners[oi];
          const vec_t* rv = reinterpret_cast<const vec_t*>(
              base + static_cast<int64_t>(o) * hidden);
          vec_t raw = rv[i];
          if constexpr (sizeof(scalar_t) == 2) {
            auto bv = *reinterpret_cast<
                sycl::vec<sycl::ext::oneapi::bfloat16, VEC_SIZE>*>(&raw);
            acc += bv.template convert<float>();
          } else {
            acc += *reinterpret_cast<sycl::vec<float, VEC_SIZE>*>(&raw);
          }
        }
        vec_t ov;
        if constexpr (sizeof(scalar_t) == 2) {
          auto bo = acc.template convert<sycl::ext::oneapi::bfloat16>();
          ov = *reinterpret_cast<vec_t*>(&bo);
        } else {
          ov = *reinterpret_cast<vec_t*>(&acc);
        }
        ovec[i] = ov;
      }
    }
  }
};

}  // namespace

// recv_buffers[rank] MUST equal recv_local; peers push into it directly. The
// caller issues the leading symmetric-memory barrier before this op. The reduce
// sums only each token's DISTINCT owner slots, so the recv buffer does NOT need
// to be re-zeroed between calls (never-written slots are simply not read).
at::Tensor ep_combine_no_permute(
    const at::Tensor& contributions,
    const at::Tensor& topk_idx_full,
    const at::Tensor& recv_buffers,
    const at::Tensor& signal_pads,
    at::Tensor recv_local,
    at::Tensor output,
    int64_t rank,
    int64_t world_size,
    int64_t num_experts,
    int64_t iteration) {
  TORCH_CHECK(
      contributions.dim() == 2 && contributions.is_contiguous(),
      "ep_combine_no_permute: contributions must be contiguous [W*T, H]");
  TORCH_CHECK(
      topk_idx_full.dim() == 2 && topk_idx_full.is_contiguous() &&
          topk_idx_full.scalar_type() == at::kInt,
      "ep_combine_no_permute: topk_idx_full must be int32 [W*T, topk]");
  TORCH_CHECK(
      topk_idx_full.size(0) == contributions.size(0),
      "ep_combine_no_permute: topk_idx_full rows must equal W*T");
  TORCH_CHECK(
      recv_local.dim() == 2 && recv_local.is_contiguous(),
      "ep_combine_no_permute: recv_local must be contiguous [T*W, H]");
  TORCH_CHECK(
      output.dim() == 2 && output.is_contiguous(),
      "ep_combine_no_permute: output must be contiguous [T, H]");
  TORCH_CHECK(
      recv_buffers.dim() == 1 && recv_buffers.size(0) == world_size &&
          recv_buffers.scalar_type() == at::kLong,
      "ep_combine_no_permute: recv_buffers must be int64[world_size]");
  TORCH_CHECK(
      signal_pads.dim() == 1 && signal_pads.size(0) == world_size &&
          signal_pads.scalar_type() == at::kLong,
      "ep_combine_no_permute: signal_pads must be int64[world_size]");
  TORCH_CHECK(
      contributions.size(0) % world_size == 0,
      "ep_combine_no_permute: rows must be divisible by world_size");
  TORCH_CHECK(
      recv_local.size(0) == contributions.size(0) &&
          recv_local.size(1) == contributions.size(1),
      "ep_combine_no_permute: recv_local shape mismatch");
  TORCH_CHECK(rank >= 0 && rank < world_size);
  TORCH_CHECK(iteration > 0, "ep_combine_no_permute: iteration must be > 0");
  TORCH_CHECK(
      contributions.scalar_type() == recv_local.scalar_type() &&
          contributions.scalar_type() == output.scalar_type(),
      "ep_combine_no_permute: dtype mismatch");

  const int64_t total_rows = contributions.size(0);
  const int64_t tokens = total_rows / world_size;
  const int64_t hidden = contributions.size(1);
  const int64_t topk = topk_idx_full.size(1);
  TORCH_CHECK(
      topk <= EPC_NP_MAX_TOPK,
      "ep_combine_no_permute: topk exceeds EPC_NP_MAX_TOPK");
  TORCH_CHECK(
      output.size(0) == tokens && output.size(1) == hidden,
      "ep_combine_no_permute: output shape must be [T, H]");
  constexpr int VEC_SIZE = 8;
  TORCH_CHECK(
      hidden % VEC_SIZE == 0,
      "ep_combine_no_permute: hidden must be divisible by 8");
  if (total_rows == 0 || hidden == 0) {
    return output;
  }

  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  // This rank's owned experts are a contiguous [expert_lo, expert_hi) range
  // (inverse of owner_of); the push path uses this to test ownership with two
  // comparisons instead of two integer divisions per (source, token, k).
  int32_t expert_lo;
  int32_t expert_hi;
  if (static_cast<int32_t>(rank) < rem_experts) {
    expert_lo = static_cast<int32_t>(rank) * (base_experts + 1);
    expert_hi = expert_lo + (base_experts + 1);
  } else {
    expert_lo = boundary + (static_cast<int32_t>(rank) - rem_experts) * base_experts;
    expert_hi = expert_lo + base_experts;
  }
  // One work-group per token slice; keep num_wg <= EPC_NP_MAX_WG so num_wg *
  // world_size fits the signal-pad region (world_size * EPC_NP_MAX_WG).
  int64_t tokens_per_wg = (tokens + EPC_NP_MAX_WG - 1) / EPC_NP_MAX_WG;
  if (tokens_per_wg < 1) {
    tokens_per_wg = 1;
  }
  if (const char* v = std::getenv("EP_COMBINE_NP_TOKENS_PER_WG")) {
    const int64_t req = std::atoi(v);
    if (req >= 1 && req <= tokens) {
      tokens_per_wg = req;
    }
  }
  int64_t num_wg = (tokens + tokens_per_wg - 1) / tokens_per_wg;
  TORCH_CHECK(
      num_wg <= EPC_NP_MAX_WG,
      "ep_combine_no_permute: num_wg exceeds signal-pad capacity");
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, recv_local.device().index());
  c10::DeviceGuard guard(device);
  auto& queue = at::xpu::getCurrentXPUStream().queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      recv_local.scalar_type(), "ep_combine_no_permute", [&]() {
        auto kernel = EpCombineNoPermuteKernel<scalar_t, VEC_SIZE>{
            contributions.data_ptr<scalar_t>(),
            topk_idx_full.data_ptr<int32_t>(),
            recv_buffers.data_ptr<int64_t>(),
            signal_pads.data_ptr<int64_t>(),
            recv_local.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int32_t>(tokens),
            static_cast<int32_t>(hidden),
            static_cast<int32_t>(hidden / VEC_SIZE),
            static_cast<int32_t>(topk),
            static_cast<int32_t>(tokens_per_wg),
            static_cast<int32_t>(rank),
            static_cast<int32_t>(world_size),
            static_cast<int32_t>(num_wg),
            static_cast<uint32_t>(iteration),
            base_experts,
            rem_experts,
            boundary,
            expert_lo,
            expert_hi};
        sycl_kernel_submit(
            sycl::range<1>(static_cast<size_t>(num_wg) * threads),
            sycl::range<1>(threads),
            queue,
            kernel);
      });
  return output;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ep_combine_no_permute(Tensor contributions, Tensor topk_idx_full, "
      "Tensor recv_buffers, Tensor signal_pads, Tensor(a!) recv_local, "
      "Tensor(a!) output, int rank, int world_size, int num_experts, "
      "int iteration) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ep_combine_no_permute", ep_combine_no_permute);
}
