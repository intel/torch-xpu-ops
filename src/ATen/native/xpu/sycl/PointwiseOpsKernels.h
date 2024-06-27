#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void addcmul_kernel(TensorIterator& iter, Scalar value);

void addcdiv_kernel(TensorIterator& iter, Scalar value);

} // namespace at::native::xpu
