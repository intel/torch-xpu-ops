#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void where_kernel(TensorIterator& iter);

void clamp_kernel(TensorIteratorBase& iter);

void clamp_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& min,
    const Scalar& max);

void clamp_min_scalar_kernel(TensorIteratorBase& iter, Scalar min);

void clamp_max_scalar_kernel(TensorIteratorBase& iter, Scalar max);

void isin_kernel(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out);

} // namespace at::native::xpu
