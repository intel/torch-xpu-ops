#pragma once
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

void flip_kernel(TensorIterator& iter, bool quantized);

void roll_kernel(
    const Tensor& input,
    Tensor& output,
    IntArrayRef shifts,
    IntArrayRef dims);

} // namespace at::native::xpu
