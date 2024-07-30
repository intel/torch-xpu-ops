#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void mse_kernel(TensorIteratorBase& iter);

void smooth_l1_kernel(TensorIteratorBase& iter, double beta);

void huber_kernel(TensorIterator& iter, double delta);

} // namespace at::native::xpu
