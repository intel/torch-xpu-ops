#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void complex_kernel(TensorIterator& iter);

void polar_kernel(TensorIterator& iter);

} // namespace at::native::xpu
