#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void digamma_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void polygamma_kernel(TensorIteratorBase& iter, int64_t n);

TORCH_XPU_API void lgamma_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
