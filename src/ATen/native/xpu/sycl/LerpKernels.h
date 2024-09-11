#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void lerp_tensor_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void lerp_scalar_kernel(
    TensorIteratorBase& iter,
    const c10::Scalar& weight);

} // namespace at::native::xpu
