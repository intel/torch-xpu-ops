#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void softshrink_kernel(
    TensorIteratorBase& iter,
    const Scalar& value);

TORCH_XPU_API void softshrink_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& value);

} // namespace at::native::xpu
