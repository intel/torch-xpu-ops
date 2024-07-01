#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void mse_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
