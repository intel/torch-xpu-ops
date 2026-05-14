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

// TP+EP owner-based dispatch kernel.
//
// Single kernel launch processes all (global_token, k, h) work items.
// Each thread:
//   1. Computes (global_token_idx, k, h) from its global index
//   2. Looks up expert from topk_idx, computes expert owner rank
//   3. Skips if expert is NOT owned by this rank (no remote read)
//   4. Otherwise: determines source rank from global_token_idx,
//      reads the token directly from that rank's symmetric memory
//      via the pointer table, and writes to remap_hidden_states

template <typename T>
struct EpDispatchKernel {
  const int64_t* rank_ptrs;      // [world_size] device ptrs to each rank's buffer
  const int64_t* topk_idx_ptr;   // [num_tokens, topk]
  T* remap_ptr;                  // [num_tokens * topk, hidden_size]
  int64_t num_tokens_per_rank;
  int64_t hidden_size;
  int64_t topk;
  int64_t num_experts;
  int64_t rank;
  int64_t world_size;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    const int64_t num_tokens = num_tokens_per_rank * world_size;
    const int64_t total = num_tokens * topk * hidden_size;
    if (idx >= total) return;

    // Decompose flat index → (global_token_idx, k, h)
    const int64_t h = idx % hidden_size;
    const int64_t t0 = idx / hidden_size;
    const int64_t k = t0 % topk;
    const int64_t global_token_idx = t0 / topk;

    // Expert ownership check (handles remainder distribution)
    const int64_t expert = topk_idx_ptr[global_token_idx * topk + k];
    const int64_t base = num_experts / world_size;
    const int64_t rem = num_experts % world_size;
    const int64_t boundary = rem * (base + 1);
    int64_t owner;
    if (expert < boundary) {
      owner = expert / (base + 1);
    } else {
      owner = rem + (expert - boundary) / base;
    }

    // Only process tokens whose expert is owned by this rank
    if (owner != rank) return;

    // Determine source rank and local token index
    const int64_t src_rank = global_token_idx / num_tokens_per_rank;
    const int64_t local_token_idx = global_token_idx % num_tokens_per_rank;

    // Selective read: directly access source rank's symmetric memory
    const T* src = reinterpret_cast<const T*>(rank_ptrs[src_rank]);
    const int64_t dst_row = global_token_idx * topk + k;
    remap_ptr[dst_row * hidden_size + h] =
        src[local_token_idx * hidden_size + h];
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

  const int64_t total = num_tokens * topk * hidden_size;
  if (total == 0) {
    return remap_hidden_states;
  }

  constexpr int64_t threads = 256;
  const int64_t blocks = (total + threads - 1) / threads;

  c10::Device device(c10::DeviceType::XPU, remap_hidden_states.device().index());
  c10::DeviceGuard guard(device);
  auto stream = at::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  AT_DISPATCH_FLOAT_AND_BFLOAT16(
      remap_hidden_states.scalar_type(), "ep_dispatch", [&]() {
        auto kfn = EpDispatchKernel<scalar_t>{
            rank_buffers_ptr.data_ptr<int64_t>(),
            topk_idx.data_ptr<int64_t>(),
            remap_hidden_states.data_ptr<scalar_t>(),
            num_tokens_per_rank,
            hidden_size,
            topk,
            num_experts,
            rank,
            world_size};
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
      "ep_dispatch(Tensor rank_buffers_ptr, Tensor topk_idx, "
      "Tensor(a!) remap_hidden_states, int num_experts, "
      "int rank, int world_size) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(symm_mem, XPU, m) {
  m.impl("ep_dispatch", ep_dispatch);
}
