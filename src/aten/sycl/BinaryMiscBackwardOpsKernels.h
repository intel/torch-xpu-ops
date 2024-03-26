#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void tanh_backward_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
