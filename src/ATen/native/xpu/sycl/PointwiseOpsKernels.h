#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void addcmul_kernel(TensorIterator& iter, Scalar value);

void addcdiv_kernel(TensorIterator& iter, Scalar value);

void mse_backward_kernel(TensorIterator& iter, const Scalar& value);

void huber_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double delta);

} // namespace at::native::xpu
