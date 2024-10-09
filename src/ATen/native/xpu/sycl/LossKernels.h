#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor& binary_cross_entropy_kernel(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& loss);

TORCH_XPU_API Tensor& binary_cross_entropy_backward_kernel(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& grad_input);

TORCH_XPU_API void soft_margin_kernel(TensorIterator& iter);

TORCH_XPU_API void soft_margin_backward_kernel(
    TensorIterator& iter,
    Scalar norm);

} // namespace at::native::xpu
