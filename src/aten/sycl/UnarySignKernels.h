#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void sgn_kernel(TensorIteratorBase& iter);

void sign_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
