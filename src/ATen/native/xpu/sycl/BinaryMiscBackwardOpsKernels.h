#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void sigmoid_backward_kernel(TensorIteratorBase& iter);

void tanh_backward_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
