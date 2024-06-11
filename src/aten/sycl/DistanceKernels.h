#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {
void cdist_kernel_impl(
    Tensor& result,
    const Tensor& x1_expanded,
    const Tensor& x2_expanded,
    double p);
} // namespace at::native::xpu