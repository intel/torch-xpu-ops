#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void argmax_kernel(TensorIterator& iter);

} // namespace at::native::xpu
