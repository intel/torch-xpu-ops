#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void argmax_kernel(TensorIterator& iter);

void argmin_kernel(TensorIterator& iter);

void and_kernel(TensorIterator& iter);

void or_kernel(TensorIterator& iter);

void mean_kernel(TensorIterator& iter);

void sum_kernel(TensorIterator& iter);

void std_var_kernel(TensorIterator& iter, double correction, bool take_sqrt);

void aminmax_kernel(TensorIterator& iter);

} // namespace at::native::xpu
