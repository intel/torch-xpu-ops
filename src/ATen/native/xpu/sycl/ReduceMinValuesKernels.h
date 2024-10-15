#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void min_values_kernel(TensorIterator& iter);

TORCH_XPU_API void min_kernel(TensorIterator& iter);

TORCH_XPU_API void min_all_kernel(TensorIterator& iter);

} // namespace at::native::xpu
