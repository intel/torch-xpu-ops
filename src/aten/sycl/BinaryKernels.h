#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

void add_kernel(TensorIteratorBase& iter, const Scalar& alpha);

void sub_kernel(TensorIteratorBase& iter, const Scalar& alpha);

void mul_kernel(TensorIteratorBase& iter);

void div_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
