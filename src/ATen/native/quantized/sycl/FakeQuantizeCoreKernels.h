#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void fake_quantize_tensor_cachemask_kernel(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    float scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max);

TORCH_XPU_API void fake_quantize_tensor_cachemask_tensor_qparams_kernel(
    Tensor& output,
    Tensor& mask,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max);

TORCH_XPU_API void _fake_quantize_grad_learnable_tensor_kernel(
    TensorIterator& iter,
    float scale,
    float inv_scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor);

TORCH_XPU_API void fake_quant_per_channel_cachemask_kernel(
    TensorIterator& iter,
    TensorIterator& iter_mask,
    int64_t quant_min,
    int64_t quant_max);

TORCH_XPU_API void _fake_quantize_grad_learnable_channel_kernel(
    TensorIterator& iter,
    int64_t quant_min,
    int64_t quant_max,
    float grad_factor);

} // namespace at::native::xpu
