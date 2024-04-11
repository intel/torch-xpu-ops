#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void bitwise_and_kernel(TensorIteratorBase& iter);

void bitwise_or_kernel(TensorIteratorBase& iter);

void bitwise_xor_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
