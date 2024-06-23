#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

void elu_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale);

void elu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result);

} // namespace at::native::xpu
