#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void log_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
