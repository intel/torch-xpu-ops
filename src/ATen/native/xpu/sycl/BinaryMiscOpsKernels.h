#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void mse_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void smooth_l1_kernel(TensorIteratorBase& iter, double beta);

TORCH_XPU_API void huber_kernel(TensorIterator& iter, double delta);

} // namespace at::native::xpu
