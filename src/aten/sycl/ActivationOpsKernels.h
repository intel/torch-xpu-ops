#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void relu_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
