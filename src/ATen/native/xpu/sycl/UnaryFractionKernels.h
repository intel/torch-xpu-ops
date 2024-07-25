#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void reciprocal_kernel(TensorIteratorBase& iter);

void ceil_kernel(TensorIteratorBase& iter);

void round_kernel(TensorIteratorBase& iter);

void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals);

} // namespace at::native::xpu
