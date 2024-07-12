#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

void softshrink_kernel(TensorIteratorBase& iter, const Scalar& value);

void softshrink_backward_kernel(TensorIteratorBase& iter, const Scalar& value);

} // namespace at::native::xpu
