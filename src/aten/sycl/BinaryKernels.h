#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void add_kernel(TensorIteratorBase& iter, const Scalar& alpha);

void sub_kernel(TensorIteratorBase& iter, const Scalar& alpha);

void mul_kernel(TensorIteratorBase& iter);

void div_kernel(TensorIteratorBase& iter);

void div_true_kernel(TensorIteratorBase& iter);

void div_trunc_kernel(TensorIteratorBase& iter);

void div_floor_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
