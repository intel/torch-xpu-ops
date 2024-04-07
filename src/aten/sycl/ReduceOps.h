#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void argmax_kernel(TensorIterator& iter);

void or_kernel(TensorIterator& iter);

void mean_kernel(TensorIterator& iter);

void sum_kernel(TensorIterator& iter);

} // namespace at::native::xpu
