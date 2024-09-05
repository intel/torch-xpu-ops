#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void bitwise_and_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void bitwise_or_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void bitwise_xor_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
