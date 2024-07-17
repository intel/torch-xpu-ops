#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void log_kernel(TensorIteratorBase& iter);

void log10_kernel(TensorIteratorBase& iter);

void log1p_kernel(TensorIteratorBase& iter);

void log2_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
