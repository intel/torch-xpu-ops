#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void log_sigmoid_forward_kernel(TensorIteratorBase& iter);

void log_sigmoid_backward_kernel(TensorIterator& iter);

} // namespace at::native::xpu
