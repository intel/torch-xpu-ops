#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void hardshrink_kernel(
    TensorIteratorBase& iter,
    const Scalar& value);

} // namespace at::native::xpu
