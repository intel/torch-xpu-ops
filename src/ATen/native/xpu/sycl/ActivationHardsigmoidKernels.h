#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void hardsigmoid_kernel(TensorIteratorBase& iter);

void hardsigmoid_backward_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
