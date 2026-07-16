#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

// ===========================================================================
// MoE owner-sparse dispatch WITHOUT permute.
//
// Dedicated no-permute dispatch kernel (separate from ep_dispatch, which does
// owner routing + expert-sort permute).  This variant keeps the natural
// TOKEN-MAJOR layout -- output row for local token t of this (source) rank is
// (rank * T + t), the SAME position a full allgather would use -- but only
// PUSHES each token to the distinct owner ranks of its top-k experts, so the
// cross-device volume scales with owned data instead of the dense (W-1)*T*H of
// a full allgather.  No sorting, no top-k expansion => genuinely no permute.
//
// PUSH model (remote stores are far cheaper than remote loads on Xe-Link /
// PCIe): each rank writes its own owned tokens into every owner rank's output
// buffer.  One work-group per local token computes the distinct-owner bitmask
// (dedup over top-k) and cooperatively copies the H-vector into each owner's
// row.  Cross-rank ordering is provided by the collective workspace barrier the
// Python wrapper issues after the launch (a single scatter, no forwarding, so
// no per-step signal pads are needed).  Rows never owned by a rank are left at
// their initialized value (the wrapper zeroes the output buffer once), which is
// correct because the combine weights those tokens by 0 on this rank.
// ===========================================================================

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// The bandwidth-critical 16-byte remote store uses an Intel GPU LSC store with
// explicit cache control (L1WB_L3WB) so contiguous remote writes get combined
// through L3 into larger burst transactions. Enabled by default; define
// EP_NP_NO_LSC_STORE to fall back to the plain vectorized store.
#if !defined(EP_NP_LSC_STORE) && !defined(EP_NP_NO_LSC_STORE)
#define EP_NP_LSC_STORE 1
#endif

#if defined(EP_NP_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
typedef uint32_t ep_np_lsc_u4 __attribute__((ext_vector_type(4)));
enum EpNpLscStcc { EP_NP_LSC_STCC_L1WB_L3WB = 7 };
SYCL_EXTERNAL extern "C" void __builtin_IB_lsc_store_global_uint4(
    __attribute__((opencl_global)) ep_np_lsc_u4* base,
    int off,
    ep_np_lsc_u4 val,
    enum EpNpLscStcc cc);
#endif

namespace {

template <typename vec_t>
inline void ep_np_vec_store(vec_t* dst, vec_t vd) {
#if defined(EP_NP_LSC_STORE) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  if constexpr (sizeof(vec_t) == 16) {
    ep_np_lsc_u4 v = *reinterpret_cast<ep_np_lsc_u4*>(&vd);
    __builtin_IB_lsc_store_global_uint4(
        (__attribute__((opencl_global)) ep_np_lsc_u4*)(dst),
        0,
        v,
        EP_NP_LSC_STCC_L1WB_L3WB);
  } else {
    *dst = vd;
  }
#else
  *dst = vd;
#endif
}

template <typename scalar_t, int VEC_SIZE>
struct EpDispatchNoPermutePushKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* input_shard;  // [T, H] this rank's local tokens
  const int32_t* topk_idx;      // [T, topk] experts of this rank's local tokens
  const int64_t* rank_buffers;  // [world_size] each rank's [W*T, H] output
  int32_t tokens_per_rank;
  int32_t hidden;
  int32_t hidden_vecs;
  int32_t topk;
  int32_t tokens_per_wg;
  int32_t rank;
  int32_t world_size;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;

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
    const int32_t token0 = wg * tokens_per_wg;

    for (int32_t tt = 0; tt < tokens_per_wg; ++tt) {
      const int32_t token = token0 + tt;
      if (token >= tokens_per_rank) {
        return;
      }

      // Distinct-owner bitmask over the token's top-k experts (dedup).
      uint32_t mask = 0;
      for (int32_t k = 0; k < topk; ++k) {
        mask |= (1u << owner_of(topk_idx[token * topk + k]));
      }

      const int64_t dst_row =
          static_cast<int64_t>(rank) * tokens_per_rank + token;
      const scalar_t* src = input_shard + static_cast<int64_t>(token) * hidden;
      const vec_t* svec = reinterpret_cast<const vec_t*>(src);

      for (int32_t o = 0; o < world_size; ++o) {
        if (!(mask & (1u << o))) {
          continue;
        }
        scalar_t* dst =
            reinterpret_cast<scalar_t*>(rank_buffers[o]) + dst_row * hidden;
        vec_t* dvec = reinterpret_cast<vec_t*>(dst);
        if (o == rank) {
          for (int32_t i = lid; i < hidden_vecs; i += lsize) {
            dvec[i] = svec[i];
          }
        } else {
          for (int32_t i = lid; i < hidden_vecs; i += lsize) {
            ep_np_vec_store(&dvec[i], svec[i]);
          }
        }
      }
    }

    // Flush this work-group's remote stores to the shared coherence point so
    // the post-launch collective barrier makes them visible to the owners.
    sycl::atomic_fence(
        sycl::memory_order::release, sycl::memory_scope::system);
  }
};

}  // namespace

// output MUST equal rank_buffers[rank]; peers write into it directly.
at::Tensor ep_dispatch_no_permute(
    const at::Tensor& input_shard,
    const at::Tensor& topk_idx,
    const at::Tensor& rank_buffers,
    at::Tensor output,
    int64_t rank,
    int64_t world_size,
    int64_t num_experts) {
  TORCH_CHECK(
      input_shard.dim() == 2 && input_shard.is_contiguous(),
      "ep_dispatch_no_permute: input_shard must be contiguous [T, H]");
  TORCH_CHECK(
      topk_idx.dim() == 2 && topk_idx.is_contiguous() &&
          topk_idx.scalar_type() == at::kInt,
      "ep_dispatch_no_permute: topk_idx must be contiguous int32 [T, topk]");
  TORCH_CHECK(
      topk_idx.size(0) == input_shard.size(0),
      "ep_dispatch_no_permute: topk_idx rows must equal T");
  TORCH_CHECK(
      output.dim() == 2 && output.is_contiguous(),
      "ep_dispatch_no_permute: output must be contiguous [world*T, H]");
  TORCH_CHECK(
      rank_buffers.dim() == 1 && rank_buffers.size(0) == world_size &&
          rank_buffers.scalar_type() == at::kLong,
      "ep_dispatch_no_permute: rank_buffers must be int64[world_size]");
  TORCH_CHECK(
      output.size(0) == world_size * input_shard.size(0),
      "ep_dispatch_no_permute: output rows must equal world_size * T");
  TORCH_CHECK(
      output.size(1) == input_shard.size(1),
      "ep_dispatch_no_permute: hidden mismatch");
  TORCH_CHECK(rank >= 0 && rank < world_size);
  TORCH_CHECK(
      input_shard.scalar_type() == output.scalar_type(),
      "ep_dispatch_no_permute: input_shard and output must share dtype");

  const int64_t tokens = input_shard.size(0);
  const int64_t hidden = input_shard.size(1);
  const int64_t topk = topk_idx.size(1);
  constexpr int VEC_SIZE = 8;
  TORCH_CHECK(
      hidden % VEC_SIZE == 0,
      "ep_dispatch_no_permute: hidden must be divisible by 8");
  if (tokens == 0 || hidden == 0) {
    return output;
  }

  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  int64_t tokens_per_wg = 1;
  if (const char* v = std::getenv("EP_DISPATCH_NP_TOKENS_PER_WG")) {
    const int64_t req = std::atoi(v);
    if (req >= 1 && req <= 64) {
      tokens_per_wg = req;
    }
  }
  const int64_t num_wg = (tokens + tokens_per_wg - 1) / tokens_per_wg;
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto& queue = at::xpu::getCurrentXPUStream().queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ep_dispatch_no_permute", [&]() {
        auto kernel = EpDispatchNoPermutePushKernel<scalar_t, VEC_SIZE>{
            input_shard.data_ptr<scalar_t>(),
            topk_idx.data_ptr<int32_t>(),
            rank_buffers.data_ptr<int64_t>(),
            static_cast<int32_t>(tokens),
            static_cast<int32_t>(hidden),
            static_cast<int32_t>(hidden / VEC_SIZE),
            static_cast<int32_t>(topk),
            static_cast<int32_t>(tokens_per_wg),
            static_cast<int32_t>(rank),
            static_cast<int32_t>(world_size),
            base_experts,
            rem_experts,
            boundary};
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
      "ep_dispatch_no_permute(Tensor input_shard, Tensor topk_idx, "
      "Tensor rank_buffers, Tensor(a!) output, int rank, int world_size, "
      "int num_experts) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ep_dispatch_no_permute", ep_dispatch_no_permute);
}
