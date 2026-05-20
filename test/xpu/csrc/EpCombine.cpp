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

// TP+EP combine kernel (reverse of ep_dispatch, ring-ordered push).
//
// ep_dispatch: ring-ordered PULL from remote hidden → ownership check → write to local remap
// ep_combine:  ring-ordered PUSH: local aggregate → write to remote output buffer
//
// Algorithm:
//   For each target_rank (ring-ordered):
//     For each token belonging to target_rank:
//       Aggregate: sum topk_weights[token,k] * local_expert_output[scatter_idx[token,k]]
//                  (only for k where this rank owns the expert)
//       Push: write partial result to target_rank's receive buffer at this rank's slot
//
//   After barrier: each rank sums received contributions from all EP ranks.
//
// Ring ordering ensures that at each step, adjacent work groups write to
// DIFFERENT target ranks, spreading PCIe write traffic evenly.
//
// Decomposition: idx → (local_token_idx, step, vec_h)
//   vec_h = idx % hidden_vecs           (innermost: coalesced write)
//   step = (idx / hidden_vecs) % world_size  (interleaved: spread writes)
//   local_token_idx = idx / (hidden_vecs * world_size)
//   target_rank = (rank + step + 1) % world_size
//
// Workspace layout on each rank: [world_size, num_tokens_per_rank, hidden]
//   slot[i] = partial contribution FROM rank i for this rank's tokens.
//   rank_output_ptrs[r] points to rank r's receive buffer base.
//   This rank writes to rank_output_ptrs[target_rank] at offset
//   rank * num_tokens_per_rank * hidden.

// Scalar fallback for non-aligned hidden_size.
template <typename T>
struct EpCombineRingScalarKernel {
  const T* expert_output_ptr;       // [num_tokens * topk, hidden] local expert results
  const int64_t* rank_output_ptrs;  // [world_size] pointers to each rank's receive buffer
  const int64_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t rank;
  int64_t world_size;
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = world_size * num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t pair_idx = idx / hidden_size;
    const int64_t step = pair_idx % world_size;
    const int64_t local_token_idx = pair_idx / world_size;

    // Ring ordering: spread writes across target ranks
    const int64_t target_rank = (rank + step + 1) % world_size;
    const int64_t global_token_idx = target_rank * num_tokens_per_rank + local_token_idx;

    // Ownership pre-check: skip if this rank doesn't own any expert for this token.
    // Caller must pre-zero the receive buffer so skipped slots remain zero.
    const int64_t topk_base = global_token_idx * topk;
    bool has_owned = false;
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == static_cast<int32_t>(rank)) { has_owned = true; break; }
    }
    if (!has_owned) return;

    // Compute weighted partial sum from local expert_output (only owned experts)
    float acc = 0.0f;
    for (int64_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == static_cast<int32_t>(rank)) {
        const float weight = topk_weights_ptr[topk_base + k];
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        acc += weight * static_cast<float>(
            expert_output_ptr[static_cast<int64_t>(src_row) * hidden_size + h]);
      }
    }

    // Push to target_rank's receive buffer at this rank's slot
    // Layout: [world_size, num_tokens_per_rank, hidden]
    T* target_buf = reinterpret_cast<T*>(rank_output_ptrs[target_rank]);
    const int64_t dst_offset =
        (rank * num_tokens_per_rank + local_token_idx) * hidden_size + h;
    target_buf[dst_offset] = static_cast<T>(acc);
  }
};

// Vectorized ring-ordered push kernel.
//
// Key optimizations (mirror ep_dispatch):
// 1. Ring ordering: adjacent work groups write to DIFFERENT target ranks,
//    spreading PCIe write bandwidth across all interconnect links.
// 2. Ownership pre-check: skip aggregation + write if no expert is owned.
//    Saves both local reads from expert_output AND expensive PCIe writes.
// 3. Reads from LOCAL expert_output (fast HBM) only for owned expert slots.
// 4. Float accumulator for precision with bfloat16 data.
template <typename scalar_t, int VEC_SIZE>
struct EpCombineRingVecKernel {
  const scalar_t* expert_output_ptr;
  const int64_t* rank_output_ptrs;
  const int64_t* topk_idx_ptr;
  const int32_t* scatter_idx_ptr;
  const float* topk_weights_ptr;
  int32_t num_tokens_per_rank;
  int32_t hidden_size;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t hidden_vecs;  // hidden_size / VEC_SIZE
  int32_t base_experts;
  int32_t rem_experts;
  int32_t boundary;

  void operator()(sycl::nd_item<1> item) const {
    const int32_t idx = static_cast<int32_t>(item.get_global_id(0));
    const int32_t total = num_tokens_per_rank * world_size * hidden_vecs;
    if (idx >= total) return;

    const int32_t vec_h = idx % hidden_vecs;
    const int32_t pair_idx = idx / hidden_vecs;
    const int32_t step = pair_idx % world_size;
    const int32_t local_token_idx = pair_idx / world_size;

    // Ring ordering: spread PCIe writes across target ranks
    const int32_t target_rank = (rank + step + 1) % world_size;
    const int32_t global_token_idx = target_rank * num_tokens_per_rank + local_token_idx;
    const int32_t h_start = vec_h * VEC_SIZE;

    // Ownership pre-check: skip if this rank doesn't own any expert for this token.
    // Caller must pre-zero the receive buffer so skipped slots remain zero.
    const int64_t topk_base = static_cast<int64_t>(global_token_idx) * topk;
    bool has_owned = false;
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) { has_owned = true; break; }
    }
    if (!has_owned) return;

    // Compute weighted partial sum from LOCAL expert_output (only owned experts)
    float acc[VEC_SIZE] = {};
    for (int32_t k = 0; k < topk; ++k) {
      const int32_t expert = static_cast<int32_t>(topk_idx_ptr[topk_base + k]);
      int32_t owner;
      if (expert < boundary) {
        owner = expert / (base_experts + 1);
      } else {
        owner = rem_experts + (expert - boundary) / base_experts;
      }
      if (owner == rank) {
        const float weight = topk_weights_ptr[topk_base + k];
        const int32_t src_row = scatter_idx_ptr[topk_base + k];
        const scalar_t* src = expert_output_ptr +
            static_cast<int64_t>(src_row) * hidden_size + h_start;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          acc[i] += weight * static_cast<float>(src[i]);
        }
      }
    }

    // Push partial result to target_rank's receive buffer at this rank's slot
    // Layout: [world_size, num_tokens_per_rank, hidden]
    scalar_t* target_buf = reinterpret_cast<scalar_t*>(rank_output_ptrs[target_rank]);
    scalar_t* dst = target_buf +
        (static_cast<int64_t>(rank) * num_tokens_per_rank + local_token_idx) * hidden_size +
        h_start;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      dst[i] = static_cast<scalar_t>(acc[i]);
    }
  }
};

at::Tensor ep_combine(
    const at::Tensor& expert_output,
    const at::Tensor& rank_output_ptrs,
    const at::Tensor& topk_idx,
    const at::Tensor& scatter_idx,
    const at::Tensor& topk_weights,
    at::Tensor output,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(
      rank_output_ptrs.dim() == 1 && rank_output_ptrs.size(0) == world_size,
      "ep_combine: rank_output_ptrs must be 1D with size == world_size");
  TORCH_CHECK(
      rank_output_ptrs.scalar_type() == at::kLong,
      "ep_combine: rank_output_ptrs must be int64");
  TORCH_CHECK(
      expert_output.dim() == 2,
      "ep_combine: expert_output must be 2D [num_tokens * topk, hidden]");
  TORCH_CHECK(expert_output.is_contiguous());
  TORCH_CHECK(topk_idx.dim() == 2, "ep_combine: topk_idx must be 2D");
  TORCH_CHECK(
      topk_idx.scalar_type() == at::kLong,
      "ep_combine: topk_idx must be int64");
  TORCH_CHECK(topk_idx.is_contiguous());
  TORCH_CHECK(
      scatter_idx.dim() == 2 && scatter_idx.size(0) == topk_idx.size(0) &&
          scatter_idx.size(1) == topk_idx.size(1),
      "ep_combine: scatter_idx must be 2D with same shape as topk_idx");
  TORCH_CHECK(
      scatter_idx.scalar_type() == at::kInt,
      "ep_combine: scatter_idx must be int32");
  TORCH_CHECK(scatter_idx.is_contiguous());
  TORCH_CHECK(
      topk_weights.dim() == 2 && topk_weights.size(0) == topk_idx.size(0) &&
          topk_weights.size(1) == topk_idx.size(1),
      "ep_combine: topk_weights must be 2D with same shape as topk_idx");
  TORCH_CHECK(
      topk_weights.scalar_type() == at::kFloat,
      "ep_combine: topk_weights must be float32");
  TORCH_CHECK(topk_weights.is_contiguous());
  TORCH_CHECK(output.dim() == 2, "ep_combine: output must be 2D");
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(rank >= 0 && rank < world_size);

  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = expert_output.size(1);

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "ep_combine: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  TORCH_CHECK(
      output.size(0) == num_tokens_per_rank,
      "ep_combine: output first dim must be num_tokens_per_rank");
  TORCH_CHECK(
      output.size(1) == hidden_size,
      "ep_combine: output hidden size must match expert_output");

  if (num_tokens == 0 || topk == 0 || hidden_size == 0) {
    return output;
  }

  // Precompute ownership constants on host
  const int32_t base_experts = static_cast<int32_t>(num_experts / world_size);
  const int32_t rem_experts = static_cast<int32_t>(num_experts % world_size);
  const int32_t boundary = rem_experts * (base_experts + 1);

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  c10::Device device(c10::DeviceType::XPU, output.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      output.scalar_type(), "ep_combine", [&]() {
        if (hidden_size % VEC_SIZE == 0) {
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          const int64_t total = world_size * num_tokens_per_rank * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = EpCombineRingVecKernel<scalar_t, VEC_SIZE>{
              expert_output.data_ptr<scalar_t>(),
              rank_output_ptrs.data_ptr<int64_t>(),
              topk_idx.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              static_cast<int32_t>(num_tokens_per_rank),
              static_cast<int32_t>(hidden_size),
              static_cast<int32_t>(topk),
              static_cast<int32_t>(rank),
              static_cast<int32_t>(world_size),
              static_cast<int32_t>(hidden_vecs),
              base_experts,
              rem_experts,
              boundary};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        } else {
          const int64_t total = world_size * num_tokens_per_rank * hidden_size;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = EpCombineRingScalarKernel<scalar_t>{
              expert_output.data_ptr<scalar_t>(),
              rank_output_ptrs.data_ptr<int64_t>(),
              topk_idx.data_ptr<int64_t>(),
              scatter_idx.data_ptr<int32_t>(),
              topk_weights.data_ptr<float>(),
              num_tokens_per_rank,
              hidden_size,
              topk,
              rank,
              world_size,
              base_experts,
              rem_experts,
              boundary};
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        }
      });

  return output;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ep_combine(Tensor expert_output, Tensor rank_output_ptrs, "
      "Tensor topk_idx, Tensor scatter_idx, Tensor topk_weights, "
      "Tensor(a!) output, int num_experts, "
      "int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ep_combine", ep_combine);
}
