#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void maximum_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void minimum_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void fmax_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void fmin_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
