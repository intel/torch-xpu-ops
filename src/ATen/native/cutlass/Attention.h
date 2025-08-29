#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace cutlass_sycl{

void sdpa_backward(
    int batch_size,
    int num_head_q,
    int num_head_kv,
    int seq_len_q,
    int seq_len_kv,
    int head_dim_qk,
    int head_dim_v,
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    std::optional<at::Tensor> attn_mask,
    bool is_causal,
    double scale,
    Tensor& grad_query,
    Tensor& grad_key,
    Tensor& grad_value);

} // namespace cutlass_sycl
} // namespace native
} // namespace at