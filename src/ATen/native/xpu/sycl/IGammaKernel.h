#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void igamma_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void igammac_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
