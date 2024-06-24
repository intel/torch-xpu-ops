#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void copy_kernel(TensorIterator& iter);

} // namespace at::native::xpu
