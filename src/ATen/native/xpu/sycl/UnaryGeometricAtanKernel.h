#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void atan_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
