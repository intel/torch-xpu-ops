#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void sigmoid_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void erf_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void erfc_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void erfinv_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void exp2_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void logit_kernel(
    TensorIteratorBase& iter,
    const Scalar& eps_scalar);

} // namespace at::native::xpu
