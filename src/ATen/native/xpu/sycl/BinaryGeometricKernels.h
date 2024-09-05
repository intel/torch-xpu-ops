#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void atan2_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void hypot_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
