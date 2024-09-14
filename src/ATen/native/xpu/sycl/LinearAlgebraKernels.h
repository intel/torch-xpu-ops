#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void addr_kernel(
    TensorIterator& iter,
    const Scalar& beta,
    const Scalar& alpha);

} // namespace at::native::xpu
