#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void _transform_bias_rescale_qkv_kernel(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head,
    Tensor& q_k_v,
    int64_t B,
    int64_t T,
    int64_t D,
    int64_t dim_per_head);

} // namespace at::native::xpu
