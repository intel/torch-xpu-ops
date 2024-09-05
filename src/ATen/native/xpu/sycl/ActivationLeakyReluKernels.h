#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void leaky_relu_kernel(
    TensorIteratorBase& iter,
    const Scalar& negval_);

TORCH_XPU_API void leaky_relu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& negval_);

} // namespace at::native::xpu
