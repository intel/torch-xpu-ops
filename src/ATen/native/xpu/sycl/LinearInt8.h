#pragma once
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void linear_int8_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& scales,
    Tensor& output);

} // namespace at::native::xpu
