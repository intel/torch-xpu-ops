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

// Kernel for TP+EP owner-based dispatch
// Dispatch hidden_shard into remap_hidden_states based on topk_idx and expert ownership

template <typename T>
struct EpDispatchKernel {
  const T* hidden_shard_ptr;
  const int64_t* topk_idx_ptr;
  T* remap_hidden_states_ptr;
  const T* symm_buffer_ptr;
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t num_experts;
  int64_t rank;
  int64_t world_size;

  EpDispatchKernel(
      const T* hidden_shard_ptr_,
      const int64_t* topk_idx_ptr_,
      T* remap_hidden_states_ptr_,
      const T* symm_buffer_ptr_,
      int64_t num_tokens_per_rank_,
      int64_t hidden_size_,
      int64_t topk_,
      int64_t num_experts_,
      int64_t rank_,
      int64_t world_size_)
      : hidden_shard_ptr(hidden_shard_ptr_),
        topk_idx_ptr(topk_idx_ptr_),
        remap_hidden_states_ptr(remap_hidden_states_ptr_),
        symm_buffer_ptr(symm_buffer_ptr_),
        num_tokens_per_rank(num_tokens_per_rank_),
        hidden_size(hidden_size_),
        topk(topk_),
        num_experts(num_experts_),
        rank(rank_),
        world_size(world_size_) {}

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t total = num_tokens_per_rank * world_size * topk;
    if (idx >= total) {
      return;
    }

    const int64_t token_idx = (idx / topk) % num_tokens_per_rank;
    const int64_t k = idx % topk;
    const int64_t remote_rank = (idx / (num_tokens_per_rank * topk)) % world_size;
    const int64_t global_token_idx = remote_rank * num_tokens_per_rank + token_idx;

    const int64_t expert = topk_idx_ptr[global_token_idx * topk + k];
    const int64_t owner = expert / (num_experts / world_size); // Simplified owner calculation

    if (owner == rank) {
      const int64_t dst_row = global_token_idx * topk + k;
      const T* source_buffer = (remote_rank == rank) ? hidden_shard_ptr : symm_buffer_ptr + remote_rank * num_tokens_per_rank * hidden_size;
      for (int64_t h = 0; h < hidden_size; ++h) {
        remap_hidden_states_ptr[dst_row * hidden_size + h] =
            source_buffer[token_idx * hidden_size + h];
      }
    }
  }
};

at::Tensor ep_dispatch(
    const at::Tensor& hidden_shard,
    const at::Tensor& topk_idx,
    at::Tensor remap_hidden_states,
    int64_t num_experts,
    int64_t rank,
    int64_t world_size) {
  TORCH_CHECK(hidden_shard.dim() == 2, "ep_dispatch: hidden_shard must be 2D");
  TORCH_CHECK(topk_idx.dim() == 2, "ep_dispatch: topk_idx must be 2D");
  TORCH_CHECK(
      hidden_shard.scalar_type() == remap_hidden_states.scalar_type(),
      "ep_dispatch: hidden_shard and remap_hidden_states dtype must match");
  TORCH_CHECK(hidden_shard.is_contiguous(), "ep_dispatch: hidden_shard must be contiguous");
  TORCH_CHECK(remap_hidden_states.is_contiguous(), "ep_dispatch: remap_hidden_states must be contiguous");

  const int64_t num_tokens_per_rank = hidden_shard.size(0);
  const int64_t hidden_size = hidden_shard.size(1);
  const int64_t num_tokens = topk_idx.size(0);
  const int64_t topk = topk_idx.size(1);

  TORCH_CHECK(
      remap_hidden_states.size(0) == num_tokens * topk,
      "ep_dispatch: remap_hidden_states first dim mismatch");
  TORCH_CHECK(
      remap_hidden_states.size(1) == hidden_size,
      "ep_dispatch: remap_hidden_states hidden size mismatch");

  const int64_t total = num_tokens_per_rank * topk;
  if (total == 0) {
    return remap_hidden_states;
  }

  constexpr int64_t threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;

  c10::Device device(c10::DeviceType::XPU, hidden_shard.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      hidden_shard.scalar_type(), "ep_dispatch", [&]() {
        auto kfn = EpDispatchKernel<scalar_t>(
            hidden_shard.data_ptr<scalar_t>(),
            topk_idx.data_ptr<int64_t>(),
            remap_hidden_states.data_ptr<scalar_t>(),
            hidden_shard.data_ptr<scalar_t>(),
            num_tokens_per_rank,
            hidden_size,
            topk,
            num_experts,
            rank,
            world_size);
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
      "ep_dispatch(Tensor hidden_shard, Tensor topk_idx, Tensor(a!) remap_hidden_states, int num_experts, int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ep_dispatch", ep_dispatch);
}
