#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void logical_and_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void logical_or_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void logical_xor_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
