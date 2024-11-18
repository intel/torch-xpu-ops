#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void mse_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void smooth_l1_kernel(TensorIteratorBase& iter, double beta);

TORCH_XPU_API void huber_kernel(TensorIterator& iter, double delta);

TORCH_XPU_API void xlogy_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void xlog1py_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
