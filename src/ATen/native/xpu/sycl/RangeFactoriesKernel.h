#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

Tensor& arange_kernel(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result);

Tensor& range_kernel(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result);

Tensor& linspace_kernel(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    Tensor& result);

} // namespace at::native::xpu
