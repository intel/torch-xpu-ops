#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {
void acos_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu