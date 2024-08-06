#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void sigmoid_backward_kernel(TensorIteratorBase& iter);

void tanh_backward_kernel(TensorIteratorBase& iter);

void logit_backward_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar);

} // namespace at::native::xpu
