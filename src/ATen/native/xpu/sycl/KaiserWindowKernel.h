#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void kaiser_window_kernel(TensorIteratorBase& iter, int64_t window_length, double beta);

} // namespace at::native::xpu
