#pragma once
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void linear_int4_kernel(
    const Tensor& input,
    const Tensor& weight,
    int qGroupSize,
    const Tensor& weight_scale_zero_point,
    Tensor& output);

} // namespace at::native::xpu
