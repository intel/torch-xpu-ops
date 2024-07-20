#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void sigmoid_kernel(TensorIteratorBase& iter);

void erf_kernel(TensorIteratorBase& iter);

void erfc_kernel(TensorIteratorBase& iter);

void logit_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar);

} // namespace at::native::xpu
