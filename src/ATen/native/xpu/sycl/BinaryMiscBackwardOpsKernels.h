#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void sigmoid_backward_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void tanh_backward_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void logit_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& eps_scalar);

} // namespace at::native::xpu
