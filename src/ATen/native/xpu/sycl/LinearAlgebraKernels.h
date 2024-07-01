#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

namespace at::native::xpu {

void addr_kernel(TensorIterator& iter, const Scalar& beta, const Scalar& alpha);

} // namespace at::native::xpu
