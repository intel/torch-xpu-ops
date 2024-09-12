#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void max_values_kernel(TensorIterator& iter);

TORCH_XPU_API void max_kernel(TensorIterator& iter);

TORCH_XPU_API void max_all_kernel(TensorIterator& iter);

} // namespace at::native::xpu
