#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void tan_kernel(TensorIterator& iter);

} // namespace at::native::xpu
