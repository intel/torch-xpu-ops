#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void mse_kernel(TensorIteratorBase& iter);

void huber_kernel(TensorIterator& iter, double delta);

} // namespace at::native::xpu
