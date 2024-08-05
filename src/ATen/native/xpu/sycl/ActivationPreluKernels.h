#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void prelu_kernel(TensorIteratorBase& iter);

void prelu_backward_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
