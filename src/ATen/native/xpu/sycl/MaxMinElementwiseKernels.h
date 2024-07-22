#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void maximum_kernel(TensorIteratorBase& iter);

void minimum_kernel(TensorIteratorBase& iter);

void fmax_kernel(TensorIteratorBase& iter);

void fmin_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
