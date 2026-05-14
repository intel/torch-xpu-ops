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

// TP+EP owner-based dispatch kernel (optimized).
//
// Improvements over the original scalar-per-thread kernel:
// 1. Vectorized load/store (VEC_SIZE=8 elements per thread)
// 2. int32 arithmetic to avoid expensive 64-bit div/mod on GPU
// 3. Precomputed ownership constants (base/rem/boundary)
// 4. Scalar fallback for non-aligned hidden_size

// Scalar fallback kernel for non-aligned hidden_size.
// Uses int64 for safety, but benefits from precomputed ownership constants.
template <typename T>
struct EpDispatchScalarKernel {
  const int64_t* rank_ptrs;
  const int64_t* topk_idx_ptr;
  T* remap_ptr;
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
    const int64_t num_tokens = num_tokens_per_rank * world_size;
    const int64_t total = num_tokens * topk * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t t0 = idx / hidden_size;
    const int64_t k = t0 % topk;
    const int64_t global_token_idx = t0 / topk;

    const int32_t expert = static_cast<int32_t>(
        topk_idx_ptr[global_token_idx * topk + k]);
    int32_t owner;
    if (expert < boundary) {
      owner = expert / (base_experts + 1);
    } else {
      owner = rem_experts + (expert - boundary) / base_experts;
    }

    if (owner != static_cast<int32_t>(rank)) return;

    const int64_t src_rank = global_token_idx / num_tokens_per_rank;
    const int64_t local_token_idx = global_token_idx % num_tokens_per_rank;

    const T* src = reinterpret_cast<const T*>(rank_ptrs[src_rank]);
    const int64_t dst_row = global_token_idx * topk + k;
    remap_ptr[dst_row * hidden_size + h] =
        src[local_token_idx * hidden_size + h];
  }
};

// Vectorized kernel: each work item loads VEC_SIZE contiguous elements from
// the source rank's symmetric memory and writes them to remap_hidden_states.
// Reduces total work items by VEC_SIZE× and improves memory throughput via
// wider load/store transactions.
template <typename scalar_t, int VEC_SIZE>
struct EpDispatchVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const int64_t* rank_ptrs;
  const int64_t* topk_idx_ptr;
  scalar_t* remap_ptr;
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
    const int32_t num_tokens = num_tokens_per_rank * world_size;
    const int32_t total = num_tokens * topk * hidden_vecs;
    if (idx >= total) return;

    // Adjacent threads handle adjacent vec positions → coalesced access
    const int32_t vec_h = idx % hidden_vecs;
    const int32_t token_k = idx / hidden_vecs;
    const int32_t k = token_k % topk;
    const int32_t global_token_idx = token_k / topk;

    // Expert ownership check with precomputed constants
    const int32_t expert = static_cast<int32_t>(
        topk_idx_ptr[global_token_idx * topk + k]);
    int32_t owner;
    if (expert < boundary) {
      owner = expert / (base_experts + 1);
    } else {
      owner = rem_experts + (expert - boundary) / base_experts;
    }

    if (owner != rank) return;

    const int32_t src_rank = global_token_idx / num_tokens_per_rank;
    const int32_t local_token_idx = global_token_idx % num_tokens_per_rank;

    // Vectorized read from source rank's symmetric memory
    const scalar_t* src = reinterpret_cast<const scalar_t*>(rank_ptrs[src_rank]);
    auto src_vec = reinterpret_cast<const vec_t*>(
        src + static_cast<int64_t>(local_token_idx) * hidden_size);
    vec_t v = src_vec[vec_h];

    // Vectorized write to remap
    const int32_t dst_row = global_token_idx * topk + k;
    auto dst_vec = reinterpret_cast<vec_t*>(
        remap_ptr + static_cast<int64_t>(dst_row) * hidden_size);
    dst_vec[vec_h] = v;
  }
};

at::Tensor ep_dispatch(
    const at::Tensor& rank_buffers_ptr,
    const at::Tensor& topk_idx,
    at::Tensor remap_hidden_states,
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
      remap_hidden_states.dim() == 2,
      "ep_dispatch: remap_hidden_states must be 2D");
  TORCH_CHECK(
      remap_hidden_states.is_contiguous(),
      "ep_dispatch: remap_hidden_states must be contiguous");

  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);
  const int64_t hidden_size = remap_hidden_states.size(1);

  TORCH_CHECK(
      num_tokens % world_size == 0,
      "ep_dispatch: num_tokens must be divisible by world_size");
  const int64_t num_tokens_per_rank = num_tokens / world_size;

  TORCH_CHECK(
      remap_hidden_states.size(0) == num_tokens * topk,
      "ep_dispatch: remap_hidden_states first dim mismatch");

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

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      remap_hidden_states.scalar_type(), "ep_dispatch", [&]() {
        if (hidden_size % VEC_SIZE == 0) {
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          const int64_t total = num_tokens * topk * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = EpDispatchVecKernel<scalar_t, VEC_SIZE>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              topk_idx.data_ptr<int64_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
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
          const int64_t blocks = (total_elems + threads - 1) / threads;
          auto kfn = EpDispatchScalarKernel<scalar_t>{
              rank_buffers_ptr.data_ptr<int64_t>(),
              topk_idx.data_ptr<int64_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
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

  return remap_hidden_states;
}

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  m.def(
      "ep_dispatch(Tensor rank_buffers_ptr, Tensor topk_idx, "
      "Tensor(a!) remap_hidden_states, int num_experts, "
      "int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ep_dispatch", ep_dispatch);
}
