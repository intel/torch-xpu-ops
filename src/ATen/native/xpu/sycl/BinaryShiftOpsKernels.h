#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void lshift_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void rshift_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
