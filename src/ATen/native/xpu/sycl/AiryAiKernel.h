#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void airy_ai_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu