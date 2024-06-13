#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

namespace at::native::xpu {
void unfold_backward_kernel(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step);
} // namespace at::native::xpu