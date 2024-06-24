#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void silu_kernel(TensorIteratorBase& iter);

void silu_backward_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
