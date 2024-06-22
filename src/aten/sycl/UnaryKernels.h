#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void sqrt_kernel(TensorIteratorBase& iter);

void rsqrt_kernel(TensorIteratorBase& iter);

void bitwise_not_kernel(TensorIteratorBase& iter);

void exp_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
