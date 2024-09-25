#pragma once

#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>

namespace at::native {

TORCH_XPU_API QuantizerPtr make_per_channel_affine_quantizer_xpu(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type);

TORCH_XPU_API QuantizerPtr make_per_tensor_affine_quantizer_xpu(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type);

} // namespace at::native
