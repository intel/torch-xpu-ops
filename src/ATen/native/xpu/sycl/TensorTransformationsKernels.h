#pragma once
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void flip_kernel(TensorIterator& iter, bool quantized);

TORCH_XPU_API void roll_kernel(
    const Tensor& input,
    Tensor& output,
    IntArrayRef shifts,
    IntArrayRef dims);

} // namespace at::native::xpu
