#pragma once

#include <ATen/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void elu_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale);

TORCH_XPU_API void elu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result);

} // namespace at::native::xpu
