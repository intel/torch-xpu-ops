#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void logical_and_kernel(TensorIterator& iter);

void logical_or_kernel(TensorIterator& iter);

void logical_xor_kernel(TensorIterator& iter);

} // namespace at::native::xpu
