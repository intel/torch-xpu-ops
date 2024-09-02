#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void add_kernel(TensorIteratorBase& iter, const Scalar& alpha);

TORCH_XPU_API void sub_kernel(TensorIteratorBase& iter, const Scalar& alpha);

TORCH_XPU_API void mul_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void div_true_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void div_trunc_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void div_floor_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
