#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void relu_kernel(TensorIteratorBase& iter);

void threshold_kernel(
    TensorIteratorBase& iter,
    const Scalar& threshold,
    const Scalar& value);

void gelu_kernel(TensorIteratorBase& iter, c10::string_view approximate);

void gelu_backward_kernel(
    TensorIteratorBase& iter,
    c10::string_view approximate);

void tanh_backward_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
