#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void prelu_kernel(TensorIterator& iter);

TORCH_XPU_API void prelu_backward_kernel(TensorIterator& iter);

} // namespace at::native::xpu
