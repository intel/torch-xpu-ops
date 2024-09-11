#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void logical_and_kernel(TensorIterator& iter);

TORCH_XPU_API void logical_or_kernel(TensorIterator& iter);

TORCH_XPU_API void logical_xor_kernel(TensorIterator& iter);

} // namespace at::native::xpu
