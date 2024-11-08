#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void quantize_tensor_per_channel_affine_kernel(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

TORCH_XPU_API void dequantize_tensor_per_channel_affine_kernel(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

TORCH_XPU_API void quantize_tensor_per_channel_float_qparams_kernel(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

TORCH_XPU_API void dequantize_tensor_per_channel_float_qparams_kernel(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

TORCH_XPU_API void quantize_tensor_per_tensor_affine_kernel(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point);

TORCH_XPU_API void dequantize_tensor_per_tensor_affine_kernel(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point);

} // namespace at::native::xpu
