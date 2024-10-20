#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void logical_not_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void neg_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void sgn_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void sign_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void signbit_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
