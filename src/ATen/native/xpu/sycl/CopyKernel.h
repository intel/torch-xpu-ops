#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void copy_kernel(TensorIterator& iter);

} // namespace at::native::xpu
