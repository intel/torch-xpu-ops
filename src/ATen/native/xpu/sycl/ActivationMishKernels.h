#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void mish_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void mish_backward_kernel(TensorIterator& iter);

} // namespace at::native::xpu
