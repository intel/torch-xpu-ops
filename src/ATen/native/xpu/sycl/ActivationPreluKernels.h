#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void prelu_kernel(TensorIterator& iter);

void prelu_backward_kernel(TensorIterator& iter);

} // namespace at::native::xpu
