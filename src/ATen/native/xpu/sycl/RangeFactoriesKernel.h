#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor& arange_kernel(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result);

TORCH_XPU_API Tensor& range_kernel(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result);

TORCH_XPU_API Tensor& linspace_kernel(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    Tensor& result);

} // namespace at::native::xpu
