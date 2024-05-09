#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void flip_xpu_kernel(TensorIterator& iter);

} // namespace at::native::xpu