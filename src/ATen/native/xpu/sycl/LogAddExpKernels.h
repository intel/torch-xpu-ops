#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

namespace at::native::xpu {

void logaddexp_kernel(TensorIteratorBase& iter);

void logaddexp2_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
