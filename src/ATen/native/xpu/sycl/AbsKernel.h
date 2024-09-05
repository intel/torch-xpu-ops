#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void abs_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
