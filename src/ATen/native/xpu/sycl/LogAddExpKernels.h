#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

namespace at::native::xpu {

TORCH_XPU_API void logaddexp_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void logaddexp2_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
