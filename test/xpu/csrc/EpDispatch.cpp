#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <comm/SYCLHelpers.h>

#ifndef AT_DISPATCH_FLOAT_AND_BFLOAT16
#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__))
#endif

// TP+EP owner-based dispatch kernel (ring-ordered, single kernel).
//
// Key optimizations:
// 1. Work items organized as (local_token_idx, step, vec_h) with steps
//    interleaved at work-group granularity. Adjacent work groups read from
//    DIFFERENT source ranks, enabling the GPU to overlap cross-device reads
//    from multiple sources concurrently within a single kernel launch.
// 2. Ring ordering: src_rank = (rank + step + 1) % world_size ensures
//    no two ranks read from the same source simultaneously.
// 3. Each work item reads source ONCE (coalesced) and loops over topk for
//    ownership check + conditional write. Amortizes cross-device read cost.
// 4. Total work items = world_size * tokens_per_rank * hidden_vecs
//    (vs old: num_tokens * topk * hidden_vecs with 75% early-exit waste).

// Scalar fallback for non-aligned hidden_size.
template <typename T, typename idx_out_t, typename weight_t>
struct EpDispatchRingScalarKernel {
  const int64_t* rank_ptrs;
  const int32_t* topk_idx_ptr;
  const int64_t* topk_weight_rank_ptrs;
  const int32_t* scatter_idx_ptr;
  T* remap_ptr;
  idx_out_t* recv_topk_idx_ptr;
  weight_t* recv_topk_weights_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t topk_weight_stride;
  int64_t rank;
  int64_t world_size;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;
  int64_t remap_rows;
  int64_t recv_idx_cols;
  int64_t recv_weight_cols;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t pair_idx = idx / hidden_size;
    const int64_t step = pair_idx % world_size;
    const int64_t local_token_idx = pair_idx / world_size;

    // Ring ordering: different ranks read different sources at the same step
    const int64_t src_rank = (rank + step + 1) % world_size;
    const int64_t global_token_idx = src_rank * num_tokens_per_rank + local_token_idx;

    // Read source element (coalesced within each work group)
    const T* src = reinterpret_cast<const T*>(rank_ptrs[src_rank]);
    const T val = src[local_token_idx * hidden_size + h];

    // Check ownership for each topk slot and write if owned
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[global_token_idx * topk + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == static_cast<int32_t>(rank)) {
        const int64_t dst_row = static_cast<int64_t>(
            scatter_idx_ptr[global_token_idx * topk + k]);
        if (dst_row < 0 || dst_row >= remap_rows) {
          continue;
        }
        remap_ptr[dst_row * hidden_size + h] = val;

        const weight_t* src_topk_weight =
            reinterpret_cast<const weight_t*>(topk_weight_rank_ptrs[src_rank]);
        const weight_t w = src_topk_weight[local_token_idx * topk_weight_stride + k];

        recv_topk_idx_ptr[dst_row * recv_idx_cols + k] =
            static_cast<idx_out_t>(expert);
        recv_topk_weights_ptr[dst_row * recv_weight_cols + k] = w;
      }
    }
  }
};

// Vectorized ring-ordered kernel with ownership pre-check.
//
// Key optimization: before issuing the expensive cross-device read from
// remote symmetric memory (PCIe), check if ANY of the topk experts for
// this token are owned by the current rank. If not, skip the read entirely.
// With uniform expert distribution (128 experts, 4 ranks, topk=8),
// ~10% of tokens have no owned assignment, saving 10% of PCIe traffic.
//
// Decomposition: idx → (local_token_idx, step, vec_h)
//   vec_h = idx % hidden_vecs           (innermost: coalesced access)
//   step = (idx / hidden_vecs) % world_size  (interleaved: overlap reads)
//   local_token_idx = idx / (hidden_vecs * world_size)
template <typename scalar_t, typename idx_out_t, typename weight_t, int VEC_SIZE>
struct EpDispatchRingVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const int64_t* rank_ptrs;
  const int32_t* topk_idx_ptr;
  const int64_t* topk_weight_rank_ptrs;
  const int32_t* scatter_idx_ptr;
  scalar_t* remap_ptr;
  idx_out_t* recv_topk_idx_ptr;
  weight_t* recv_topk_weights_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t topk_weight_stride;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;
  int64_t remap_rows;
  int64_t recv_idx_cols;
  int64_t recv_weight_cols;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * world_size * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t pair_idx = idx / hidden_vecs;
    const int32_t step = pair_idx % world_size;
    const int32_t local_token_idx = pair_idx / world_size;

    // Ring ordering avoids all ranks hitting the same source simultaneously
    const int32_t src_rank = (rank + step + 1) % world_size;
    const int32_t global_token_idx = src_rank * num_tokens_per_rank + local_token_idx;

    // Ownership pre-check: avoid expensive PCIe read if no expert is owned.
    // topk_idx is in local HBM (fast L2-cached read), while source may be
    // in remote symmetric memory (slow PCIe read).
    const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;
    bool has_owned = false;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) { has_owned = true; break; }
    }
    if (!has_owned) return;

    // Coalesced vectorized read from source rank's symmetric memory
    const scalar_t* src = reinterpret_cast<const scalar_t*>(rank_ptrs[src_rank]);
    auto src_vec = reinterpret_cast<const vec_t*>(
        src + static_cast<int64_t>(local_token_idx) * hidden_size);
    vec_t v = src_vec[vec_h];

    // Loop over topk: check ownership and write to owned positions
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = topk_idx_ptr[topk_base + k];
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) {
        const int32_t dst_row = scatter_idx_ptr[
            static_cast<int64_t>(global_token_idx) * topk + k];
        if (dst_row < 0 || static_cast<int64_t>(dst_row) >= remap_rows) {
          continue;
        }
        auto dst_vec = reinterpret_cast<vec_t*>(
            remap_ptr + static_cast<int64_t>(dst_row) * hidden_size);
        dst_vec[vec_h] = v;

        // write recv_topk_idx/recv_topk_weights once per (token, k)
        if (vec_h == 0) {
          const weight_t* src_topk_weight =
              reinterpret_cast<const weight_t*>(topk_weight_rank_ptrs[src_rank]);
          const weight_t w =
              src_topk_weight[static_cast<int64_t>(local_token_idx) * topk_weight_stride + k];

          recv_topk_idx_ptr[static_cast<int64_t>(dst_row) * recv_idx_cols + k] =
              static_cast<idx_out_t>(expert);
          recv_topk_weights_ptr[static_cast<int64_t>(dst_row) * recv_weight_cols + k] = w;
        }
      }
    }
  }
};

at::Tensor ep_dispatch(
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& topk_idx,
    const at::Tensor& topk_weight_rank_buffers_ptr,
    const at::Tensor& scatter_idx,
    at::Tensor remap_hidden_states,
    at::Tensor recv_topk_idx,
    at::Tensor recv_topk_weights,
    int64_t topk_weight_stride,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_buffers_ptr.dim() == 1 && rank_buffers_ptr.size(0) == world_size,
      "ep_dispatch: rank_buffers_ptr must be 1D with size == world_size");
  TORCH_CHECK(
      rank_buffers_ptr.scalar_type() == at::kLong,
      "ep_dispatch: rank_buffers_ptr must be int64");
  TORCH_CHECK(topk_idx.dim() == 2, "ep_dispatch: topk_idx must be 2D");
  TORCH_CHECK(
      topk_idx.scalar_type() == at::kInt,
      "ep_dispatch: topk_idx must be int32");
    TORCH_CHECK(
      topk_weight_rank_buffers_ptr.dim() == 1 &&
        topk_weight_rank_buffers_ptr.size(0) == world_size,
      "ep_dispatch: topk_weight_rank_buffers_ptr must be 1D with size == world_size");
    TORCH_CHECK(
      topk_weight_rank_buffers_ptr.scalar_type() == at::kLong,
      "ep_dispatch: topk_weight_rank_buffers_ptr must be int64");
  TORCH_CHECK(
      scatter_idx.dim() == 2 && scatter_idx.size(0) == topk_idx.size(0) &&
          scatter_idx.size(1) == topk_idx.size(1),
      "ep_dispatch: scatter_idx must be 2D with same shape as topk_idx");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "ep_dispatch: scatter_idx must be int32");
  TORCH_CHECK(
      remap_hidden_states.dim() == 2,
      "ep_dispatch: remap_hidden_states must be 2D");
  TORCH_CHECK(
      remap_hidden_states.is_contiguous(),
      "ep_dispatch: remap_hidden_states must be contiguous");
      TORCH_CHECK(
        recv_topk_idx.dim() == 2,
        "ep_dispatch: recv_topk_idx must be 2D [rows, topk]");
      TORCH_CHECK(
        recv_topk_weights.dim() == 2,
        "ep_dispatch: recv_topk_weights must be 2D [rows, topk]");
    TORCH_CHECK(
      recv_topk_idx.is_contiguous(),
      "ep_dispatch: recv_topk_idx must be contiguous");
    TORCH_CHECK(
      recv_topk_weights.is_contiguous(),
      "ep_dispatch: recv_topk_weights must be contiguous");
    TORCH_CHECK(
      recv_topk_weights.scalar_type() == at::kFloat ||
        recv_topk_weights.scalar_type() == at::kHalf ||
        recv_topk_weights.scalar_type() == at::kBFloat16,
      "ep_dispatch: recv_topk_weights must be float/half/bfloat16");
    TORCH_CHECK(
      recv_topk_idx.scalar_type() == at::kInt || recv_topk_idx.scalar_type() == at::kLong,
      "ep_dispatch: recv_topk_idx must be int32 or int64");

  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = remap_hidden_states.size(1);
  TORCH_CHECK(
      topk_weight_stride >= topk,
      "ep_dispatch: topk_weight_stride must be >= topk");

  TORCH_CHECK(
      recv_topk_idx.size(1) == topk,
      "ep_dispatch: recv_topk_idx second dim must equal topk");
  TORCH_CHECK(
      recv_topk_weights.size(1) == topk,
      "ep_dispatch: recv_topk_weights second dim must equal topk");
    TORCH_CHECK(
      recv_topk_weights.size(0) == recv_topk_idx.size(0),
      "ep_dispatch: recv_topk_weights and recv_topk_idx must have same rows");

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "ep_dispatch: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  TORCH_CHECK(
      remap_hidden_states.size(0) == num_tokens * topk,
      "ep_dispatch: remap_hidden_states first dim mismatch");
    TORCH_CHECK(
      recv_topk_idx.size(0) == remap_hidden_states.size(0),
      "ep_dispatch: recv_topk_idx rows must equal remap_hidden_states rows");
    TORCH_CHECK(
      recv_topk_weights.size(0) == remap_hidden_states.size(0),
      "ep_dispatch: recv_topk_weights rows must equal remap_hidden_states rows");

  const int64_t total_elems = num_tokens * topk * hidden_size;
  if (total_elems == 0) {
    return remap_hidden_states;
  }

  // Precompute ownership constants on host
  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, remap_hidden_states.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  const int64_t recv_idx_cols = recv_topk_idx.size(1);
  const int64_t recv_weight_cols = recv_topk_weights.size(1);
  const int64_t remap_rows = remap_hidden_states.size(0);

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      remap_hidden_states.scalar_type(), "ep_dispatch", [&]() {
        using remap_t = scalar_t;
        AT_DISPATCH_SWITCH(recv_topk_idx.scalar_type(), "ep_dispatch_recv_idx",
          AT_DISPATCH_CASE(at::kInt, [&]() {
            using idx_out_t = int32_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf,
                at::kBFloat16,
                recv_topk_weights.scalar_type(),
                "ep_dispatch_recv_weight",
                [&]() {
                  using weight_t = scalar_t;
                  if (hidden_size % VEC_SIZE == 0) {
                    const int64_t hidden_vecs = hidden_size / VEC_SIZE;
                    const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
                    const int64_t blocks = (total + threads - 1) / threads;
                    auto kfn = EpDispatchRingVecKernel<remap_t, idx_out_t, weight_t, VEC_SIZE>{
                        rank_buffers_ptr.data_ptr<int64_t>(),
                        topk_idx.data_ptr<int32_t>(),
                        topk_weight_rank_buffers_ptr.data_ptr<int64_t>(),
                        scatter_idx.data_ptr<int32_t>(),
                        remap_hidden_states.data_ptr<remap_t>(),
                        recv_topk_idx.data_ptr<idx_out_t>(),
                        recv_topk_weights.data_ptr<weight_t>(),
                        static_cast<int32_t>(num_tokens_per_rank),
                        static_cast<int32_t>(hidden_size),
                        static_cast<int32_t>(topk),
                        static_cast<int32_t>(topk_weight_stride),
                        static_cast<int32_t>(rank),
                        static_cast<int32_t>(world_size),
                        static_cast<int32_t>(hidden_vecs),
                        base_experts,
                        rem_experts,
                        boundary,
                        remap_rows,
                        recv_idx_cols,
                        recv_weight_cols};
                    sycl_kernel_submit(
                        sycl::range<1>(blocks * threads),
                        sycl::range<1>(threads),
                        queue,
                        kfn);
                  } else {
                    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
                    const int64_t blocks = (total + threads - 1) / threads;
                    auto kfn = EpDispatchRingScalarKernel<remap_t, idx_out_t, weight_t>{
                        rank_buffers_ptr.data_ptr<int64_t>(),
                        topk_idx.data_ptr<int32_t>(),
                        topk_weight_rank_buffers_ptr.data_ptr<int64_t>(),
                        scatter_idx.data_ptr<int32_t>(),
                        remap_hidden_states.data_ptr<remap_t>(),
                        recv_topk_idx.data_ptr<idx_out_t>(),
                        recv_topk_weights.data_ptr<weight_t>(),
                        num_tokens_per_rank,
                        hidden_size,
                        topk,
                        topk_weight_stride,
                        rank,
                        world_size,
                        base_experts,
                        rem_experts,
                        boundary,
                        remap_rows,
                        recv_idx_cols,
                        recv_weight_cols};
                    sycl_kernel_submit(
                        sycl::range<1>(blocks * threads),
                        sycl::range<1>(threads),
                        queue,
                        kfn);
                  }
                });
          });
          AT_DISPATCH_CASE(at::kLong, [&]() {
            using idx_out_t = int64_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf,
                at::kBFloat16,
                recv_topk_weights.scalar_type(),
                "ep_dispatch_recv_weight",
                [&]() {
                  using weight_t = scalar_t;
                  if (hidden_size % VEC_SIZE == 0) {
                    const int64_t hidden_vecs = hidden_size / VEC_SIZE;
                    const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
                    const int64_t blocks = (total + threads - 1) / threads;
                    auto kfn = EpDispatchRingVecKernel<remap_t, idx_out_t, weight_t, VEC_SIZE>{
                        rank_buffers_ptr.data_ptr<int64_t>(),
                        topk_idx.data_ptr<int32_t>(),
                        topk_weight_rank_buffers_ptr.data_ptr<int64_t>(),
                        scatter_idx.data_ptr<int32_t>(),
                        remap_hidden_states.data_ptr<remap_t>(),
                        recv_topk_idx.data_ptr<idx_out_t>(),
                        recv_topk_weights.data_ptr<weight_t>(),
                        static_cast<int32_t>(num_tokens_per_rank),
                        static_cast<int32_t>(hidden_size),
                        static_cast<int32_t>(topk),
                        static_cast<int32_t>(topk_weight_stride),
                        static_cast<int32_t>(rank),
                        static_cast<int32_t>(world_size),
                        static_cast<int32_t>(hidden_vecs),
                        base_experts,
                        rem_experts,
                        boundary,
                        remap_rows,
                        recv_idx_cols,
                        recv_weight_cols};
                    sycl_kernel_submit(
                        sycl::range<1>(blocks * threads),
                        sycl::range<1>(threads),
                        queue,
                        kfn);
                  } else {
                    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
                    const int64_t blocks = (total + threads - 1) / threads;
                    auto kfn = EpDispatchRingScalarKernel<remap_t, idx_out_t, weight_t>{
                        rank_buffers_ptr.data_ptr<int64_t>(),
                        topk_idx.data_ptr<int32_t>(),
                        topk_weight_rank_buffers_ptr.data_ptr<int64_t>(),
                        scatter_idx.data_ptr<int32_t>(),
                        remap_hidden_states.data_ptr<remap_t>(),
                        recv_topk_idx.data_ptr<idx_out_t>(),
                        recv_topk_weights.data_ptr<weight_t>(),
                        num_tokens_per_rank,
                        hidden_size,
                        topk,
                        topk_weight_stride,
                        rank,
                        world_size,
                        base_experts,
                        rem_experts,
                        boundary,
                        remap_rows,
                        recv_idx_cols,
                        recv_weight_cols};
                    sycl_kernel_submit(
                        sycl::range<1>(blocks * threads),
                        sycl::range<1>(threads),
                        queue,
                        kfn);
                  }
                });
          }));
      });

  return remap_hidden_states;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ep_dispatch(Tensor rank_buffers_ptr, Tensor topk_idx, "
      "Tensor topk_weight_rank_buffers_ptr, Tensor scatter_idx, "
      "Tensor(a!) remap_hidden_states, Tensor(a!) recv_topk_idx, Tensor(a!) recv_topk_weights, "
      "int topk_weight_stride, int num_experts, "
      "int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ep_dispatch", ep_dispatch);
}
