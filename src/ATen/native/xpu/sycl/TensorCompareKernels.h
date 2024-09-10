#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void where_kernel(TensorIterator& iter);

TORCH_XPU_API void clamp_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void clamp_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& min,
    const Scalar& max);

TORCH_XPU_API void clamp_min_scalar_kernel(
    TensorIteratorBase& iter,
    Scalar min);

TORCH_XPU_API void clamp_max_scalar_kernel(
    TensorIteratorBase& iter,
    Scalar max);

TORCH_XPU_API void isin_kernel(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out);

} // namespace at::native::xpu
