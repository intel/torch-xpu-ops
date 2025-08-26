#include <ATen/core/Tensor.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#endif

#include <ATen/native/cutlass/Attention.h>
#include <ATen/native/cutlass/sycl/AttentionKernels.h>

#include <comm/SYCLContext.h>

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
    Tensor& grad_value) {
    
    std::cout << "lfq: entering cutlass sdpa_backward" << std::endl;
    
    auto ps = at::matmul(query, key.transpose(-2, -1));
    ps = ps / std::sqrt(scale);
    ps = at::softmax(ps, -1).to(query.dtype());
    auto dps = at::empty_like(ps);
    cutlass_sdpa_backward(batch_size, num_head_q, num_head_kv, seq_len_q, seq_len_kv,
                 head_dim_qk, head_dim_v,
                 grad_out.data_ptr(),
                 query.data_ptr(),
                 key.data_ptr(),
                 value.data_ptr(),
                 ps.data_ptr(),
                 nullptr,
                 grad_query.data_ptr(),
                 grad_key.data_ptr(),
                 grad_value.data_ptr(),
                 dps.data_ptr());
}
} // cutlass_sycl
} // namespace native
} // namespace at