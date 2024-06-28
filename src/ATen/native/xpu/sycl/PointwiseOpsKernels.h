#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void addcmul_kernel(TensorIteratorBase& iter, const Scalar& value);

} // namespace at::native::xpu
