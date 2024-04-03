#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void gcd_kernel_xpu(TensorIteratorBase& iter);

}
} // namespace native
} // namespace at
