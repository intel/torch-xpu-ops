#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

Tensor& arange_xpu_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result);

} // namespace xpu
} // namespace native
} // namespace at
