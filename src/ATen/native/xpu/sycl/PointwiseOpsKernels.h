#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void addcmul_kernel(
    TensorIteratorBase& iter,
    const Scalar& value);

TORCH_XPU_API void addcdiv_kernel(
    TensorIteratorBase& iter,
    const Scalar& value);

TORCH_XPU_API void mse_backward_kernel(
    TensorIterator& iter,
    const Scalar& value);

TORCH_XPU_API void smooth_l1_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double beta);

TORCH_XPU_API void huber_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double delta);

} // namespace at::native::xpu
