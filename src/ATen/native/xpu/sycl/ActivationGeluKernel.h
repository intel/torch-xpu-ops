#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void gelu_kernel(
    TensorIteratorBase& iter,
    std::string_view approximate);

TORCH_XPU_API void gelu_backward_kernel(
    TensorIteratorBase& iter,
    std::string_view approximate);

} // namespace xpu
} // namespace native
} // namespace at
