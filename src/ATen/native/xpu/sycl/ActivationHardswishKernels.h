#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void hardswish_kernel(TensorIterator& iter);

void hardswish_backward_kernel(TensorIterator& iter);

} // namespace xpu
} // namespace native
} // namespace at
