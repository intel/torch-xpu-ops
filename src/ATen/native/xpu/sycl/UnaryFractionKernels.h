#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void reciprocal_kernel(TensorIteratorBase& iter);

void floor_kernel(TensorIteratorBase& iter);

void ceil_kernel(TensorIteratorBase& iter);

void round_kernel(TensorIteratorBase& iter);

void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals);

void frac_kernel(TensorIteratorBase& iter);

void trunc_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
