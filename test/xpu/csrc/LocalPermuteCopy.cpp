#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

// Fused local permute copy: [tokens_per_rank, hidden] -> remap[token*topk + k]
template <typename T>
struct LocalPermuteCopyKernel {
  const T* src_ptr;
  T* dst_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t remote_token_offset;

  LocalPermuteCopyKernel(
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
    const int64_t total = num_tokens_per_rank * topk * hidden_size;
    if (idx >= total) {
      return;
    }
    const int64_t h = idx % hidden_size;
    const int64_t t0 = idx / hidden_size;
    const int64_t k = t0 % topk;
    const int64_t local_token_idx = t0 / topk;
    const int64_t global_token_idx = remote_token_offset + local_token_idx;
    const int64_t dst_row = global_token_idx * topk + k;
    dst_ptr[dst_row * hidden_size + h] = src_ptr[local_token_idx * hidden_size + h];
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

  const int64_t total = num_tokens_per_rank * topk * hidden_size;
  if (total == 0) {
    return remap_hidden_states;
  }

  constexpr int64_t threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;

  c10::Device device(c10::DeviceType::XPU, src_hidden.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      src_hidden.scalar_type(), "local_permute_copy_", [&]() {
        auto kfn = LocalPermuteCopyKernel<scalar_t>(
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
