#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

void softplus_kernel(
    TensorIteratorBase& iter,
    const Scalar& beta_,
    const Scalar& threshold_);

void softplus_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& beta_,
    const Scalar& threshold_);

} // namespace at::native::xpu
