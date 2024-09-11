#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void complex_kernel(TensorIterator& iter);

TORCH_XPU_API void polar_kernel(TensorIterator& iter);

} // namespace at::native::xpu
