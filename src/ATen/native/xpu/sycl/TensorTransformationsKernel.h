#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void flip_kernel(TensorIterator& iter, bool quantized);

} // namespace at::native::xpu