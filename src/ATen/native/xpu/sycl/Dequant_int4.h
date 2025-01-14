
#pragma once
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void dequant_int4_kernel(
    const Tensor& weight_int4,
    Tensor& weight,
    int qGroupSize,
    const Tensor& qScaleAndZeros);

} // namespace at::native::xpu
