#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void hardswish_kernel(TensorIterator& iter);

TORCH_XPU_API void hardswish_backward_kernel(TensorIterator& iter);

} // namespace xpu
} // namespace native
} // namespace at
