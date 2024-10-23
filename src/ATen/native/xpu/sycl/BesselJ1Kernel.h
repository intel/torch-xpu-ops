#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void bessel_j1_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
