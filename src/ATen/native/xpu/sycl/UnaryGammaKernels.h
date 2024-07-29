#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void digamma_kernel(TensorIteratorBase& iter);

void polygamma_kernel(TensorIteratorBase& iter, int64_t n);

void lgamma_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
