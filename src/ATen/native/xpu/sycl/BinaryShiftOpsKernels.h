#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void lshift_kernel(TensorIteratorBase& iter);

void rshift_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
