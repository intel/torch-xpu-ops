#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void mse_backward_kernel(TensorIterator& iter, const Scalar& value);

} // namespace xpu
} // namespace native
} // namespace at
