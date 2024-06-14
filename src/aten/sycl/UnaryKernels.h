#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void abs_kernel(TensorIteratorBase& iter);

void sin_kernel(TensorIteratorBase& iter);

void cos_kernel(TensorIteratorBase& iter);

void sqrt_kernel(TensorIteratorBase& iter);

void rsqrt_kernel(TensorIteratorBase& iter);

void tanh_kernel(TensorIteratorBase& iter);

void neg_kernel(TensorIteratorBase& iter);

void reciprocal_kernel(TensorIteratorBase& iter);

void bitwise_not_kernel(TensorIteratorBase& iter);

void exp_kernel(TensorIteratorBase& iter);

void erf_kernel(TensorIteratorBase& iter);

void erfc_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
