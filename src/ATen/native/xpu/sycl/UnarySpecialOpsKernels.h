#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void sigmoid_kernel(TensorIteratorBase& iter);

void erf_kernel(TensorIteratorBase& iter);

void erfc_kernel(TensorIteratorBase& iter);

void erfinv_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
