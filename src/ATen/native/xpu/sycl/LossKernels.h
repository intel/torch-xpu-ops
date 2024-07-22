#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

Tensor& binary_cross_entropy_kernel(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& loss);

Tensor& binary_cross_entropy_backward_kernel(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& grad_input);

} // namespace at::native::xpu
