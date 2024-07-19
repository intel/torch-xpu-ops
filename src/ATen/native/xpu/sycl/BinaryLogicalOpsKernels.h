#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void logical_and_kernel(TensorIteratorBase& iter);

void logical_or_kernel(TensorIteratorBase& iter);

void logical_xor_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
