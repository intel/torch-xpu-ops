#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void masked_fill_kernel(TensorIterator& iter, const Scalar& value);

} // namespace at::native::xpu
