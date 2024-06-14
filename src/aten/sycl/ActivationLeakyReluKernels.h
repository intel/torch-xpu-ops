#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void leaky_relu_kernel(TensorIteratorBase& iter, const Scalar& negval_);

void leaky_relu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& negval_);

} // namespace xpu
} // namespace native
} // namespace at
