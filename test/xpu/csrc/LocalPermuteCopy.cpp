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

// Scalar fallback kernel for non-aligned hidden_size
template <typename T>
struct LocalPermuteCopyScalarKernel {
  const T* src_ptr;
  T* dst_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t remote_token_offset;

  LocalPermuteCopyScalarKernel(
      const T* src_ptr_,
      T* dst_ptr_,
      int64_t num_tokens_per_rank_,
      int64_t hidden_size_,
      int64_t topk_,
      int64_t remote_token_offset_)
      : src_ptr(src_ptr_),
        dst_ptr(dst_ptr_),
        num_tokens_per_rank(num_tokens_per_rank_),
        hidden_size(hidden_size_),
        topk(topk_),
        remote_token_offset(remote_token_offset_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = num_tokens_per_rank * hidden_size;
    if (idx >= total) return;

    const int64_t h = idx % hidden_size;
    const int64_t local_token_idx = idx / hidden_size;
    const int64_t global_token_idx = remote_token_offset + local_token_idx;
    const T val = src_ptr[local_token_idx * hidden_size + h];
    for (int64_t k = 0; k < topk; ++k) {
      const int64_t dst_row = global_token_idx * topk + k;
      dst_ptr[dst_row * hidden_size + h] = val;
    }
  }
};

// Vectorized kernel: each work item loads VEC_SIZE contiguous elements from src
// once and scatters to all topk dst positions. This reduces source bandwidth by
// topk and improves memory throughput via wider load/store transactions.
template <typename scalar_t, int VEC_SIZE>
struct LocalPermuteCopyVecKernel {
  using vec_elem_t =
      std::conditional_t<sizeof(scalar_t) == 2, uint16_t, uint32_t>;
  using vec_t = sycl::vec<vec_elem_t, VEC_SIZE>;

  const scalar_t* src_ptr;
  scalar_t* dst_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t remote_token_offset;
  int64_t hidden_vecs; // hidden_size / VEC_SIZE

  LocalPermuteCopyVecKernel(
      const scalar_t* src_ptr_,
      scalar_t* dst_ptr_,
      int64_t num_tokens_per_rank_,
      int64_t hidden_size_,
      int64_t topk_,
      int64_t remote_token_offset_,
      int64_t hidden_vecs_)
      : src_ptr(src_ptr_),
        dst_ptr(dst_ptr_),
        num_tokens_per_rank(num_tokens_per_rank_),
        hidden_size(hidden_size_),
        topk(topk_),
        remote_token_offset(remote_token_offset_),
        hidden_vecs(hidden_vecs_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = num_tokens_per_rank * hidden_vecs;
    if (idx >= total) return;

    const int64_t vec_h = idx % hidden_vecs;
    const int64_t local_token_idx = idx / hidden_vecs;
    const int64_t global_token_idx = remote_token_offset + local_token_idx;

    // Load source vector once
    auto src_vec =
        reinterpret_cast<const vec_t*>(src_ptr + local_token_idx * hidden_size);
    vec_t v = src_vec[vec_h];

    // Store to all topk destination rows
    for (int64_t k = 0; k < topk; ++k) {
      const int64_t dst_row = global_token_idx * topk + k;
      auto dst_vec =
          reinterpret_cast<vec_t*>(dst_ptr + dst_row * hidden_size);
      dst_vec[vec_h] = v;
    }
  }
};

at::Tensor local_permute_copy_(
    const at::Tensor& src_hidden,
    const at::Tensor& topk_idx,
    int64_t remote_token_offset,
    at::Tensor remap_hidden_states) {
  TORCH_CHECK(src_hidden.dim() == 2, "local_permute_copy_: src_hidden must be 2D");
  TORCH_CHECK(topk_idx.dim() == 2, "local_permute_copy_: topk_idx must be 2D");
  TORCH_CHECK(
      src_hidden.scalar_type() == remap_hidden_states.scalar_type(),
      "local_permute_copy_: src and remap dtype must match");
  TORCH_CHECK(src_hidden.is_contiguous(), "local_permute_copy_: src_hidden must be contiguous");
  TORCH_CHECK(remap_hidden_states.is_contiguous(), "local_permute_copy_: remap_hidden_states must be contiguous");

  const int64_t num_tokens_per_rank = src_hidden.size(0);
  const int64_t hidden_size = src_hidden.size(1);
  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);

  TORCH_CHECK(remote_token_offset >= 0, "local_permute_copy_: remote_token_offset must be >= 0");
  TORCH_CHECK(
      remote_token_offset + num_tokens_per_rank <= num_tokens,
      "local_permute_copy_: remote token range out of bounds");
  TORCH_CHECK(
      remap_hidden_states.size(0) == num_tokens * topk,
      "local_permute_copy_: remap_hidden_states first dim mismatch");
  TORCH_CHECK(
      remap_hidden_states.size(1) == hidden_size,
      "local_permute_copy_: remap_hidden_states hidden size mismatch");

  if (num_tokens_per_rank == 0) {
    return remap_hidden_states;
  }

  c10::Device device(c10::DeviceType::XPU, src_hidden.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  constexpr int VEC_SIZE = 8;
  constexpr int64_t threads = 256;

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      src_hidden.scalar_type(), "local_permute_copy_", [&]() {
        if (hidden_size % VEC_SIZE == 0) {
          // Vectorized path: each work item copies VEC_SIZE elements,
          // loading src once and writing to all topk destinations.
          const int64_t hidden_vecs = hidden_size / VEC_SIZE;
          const int64_t total = num_tokens_per_rank * hidden_vecs;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = LocalPermuteCopyVecKernel<scalar_t, VEC_SIZE>(
              src_hidden.data_ptr<scalar_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
              num_tokens_per_rank,
              hidden_size,
              topk,
              remote_token_offset,
              hidden_vecs);
          sycl_kernel_submit(
              sycl::range<1>(blocks * threads),
              sycl::range<1>(threads),
              queue,
              kfn);
        } else {
          // Scalar fallback for non-aligned hidden_size.
          // Still loads src once per (token, h) and writes to all topk dsts.
          const int64_t total = num_tokens_per_rank * hidden_size;
          const int64_t blocks = (total + threads - 1) / threads;
          auto kfn = LocalPermuteCopyScalarKernel<scalar_t>(
              src_hidden.data_ptr<scalar_t>(),
              remap_hidden_states.data_ptr<scalar_t>(),
              num_tokens_per_rank,
              hidden_size,
              topk,
              remote_token_offset);
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
      "local_permute_copy_(Tensor src_hidden, Tensor topk_idx, int remote_token_offset, Tensor(a!) remap_hidden_states) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("local_permute_copy_", local_permute_copy_);
}
