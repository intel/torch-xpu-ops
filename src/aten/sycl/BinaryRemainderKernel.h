#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void remainder_kernel(TensorIteratorBase& iter);

void fmod_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
