#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void gcd_kernel(TensorIteratorBase& iter);

}
} // namespace native
} // namespace at
