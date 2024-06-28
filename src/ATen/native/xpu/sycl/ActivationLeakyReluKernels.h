#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_);

void leaky_relu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& negval_);

} // namespace at::native::xpu
