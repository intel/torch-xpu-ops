#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void silu_kernel(TensorIteratorBase& iter);

void silu_backward_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
