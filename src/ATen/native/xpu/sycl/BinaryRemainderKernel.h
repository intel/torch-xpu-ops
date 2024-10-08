#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void remainder_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void fmod_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
