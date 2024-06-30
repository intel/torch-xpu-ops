#pragma once
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void flip_kernel(TensorIterator& iter);

void roll_kernel(
    const Tensor& input,
    Tensor& output,
    IntArrayRef shifts,
    IntArrayRef dims);

} // namespace at::native::xpu
