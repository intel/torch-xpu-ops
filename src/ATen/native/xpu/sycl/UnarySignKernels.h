#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void logical_not_kernel(TensorIteratorBase& iter);

void neg_kernel(TensorIteratorBase& iter);

void sgn_kernel(TensorIteratorBase& iter);

void sign_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
