#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void maximum_kernel(TensorIteratorBase& iter);

void minimum_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
