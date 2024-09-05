#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

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

} // namespace xpu
} // namespace native
} // namespace at
