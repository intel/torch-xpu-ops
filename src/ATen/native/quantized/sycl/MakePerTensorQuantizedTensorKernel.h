#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void assign_quantized_tensor_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
