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

TORCH_XPU_API void i0_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void i0e_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void i1_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void i1e_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void ndtri_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void log_ndtr_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void entr_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void erfcx_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void sinc_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
