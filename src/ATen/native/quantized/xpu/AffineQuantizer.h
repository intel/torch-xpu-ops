#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/quantized/Quantizer.h>

namespace at::native {

TORCH_XPU_API Tensor quantize_tensor_per_channel_affine_xpu(
    Tensor& qtensor,
    const Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

TORCH_XPU_API Tensor dequantize_tensor_per_channel_affine_xpu(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

TORCH_XPU_API Tensor quantize_tensor_per_tensor_affine_xpu(
    Tensor& qtensor,
    const Tensor& rtensor,
    double scale,
    int64_t zero_point);

TORCH_XPU_API Tensor dequantize_tensor_per_tensor_affine_xpu(
    Tensor& rtensor,
    const Tensor& qtensor,
    double scale,
    int64_t zero_point);

} // namespace at::native
