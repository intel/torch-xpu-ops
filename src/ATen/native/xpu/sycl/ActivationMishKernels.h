#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void mish_kernel(TensorIteratorBase& iter);

void mish_backward_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
