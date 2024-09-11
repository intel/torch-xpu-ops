#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void conj_kernel(TensorIterator& iter);

TORCH_XPU_API void conj_physical_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void neg_conj_kernel(TensorIterator& iter);

TORCH_XPU_API void neg_kernel(TensorIterator& iter);

} // namespace at::native::xpu
