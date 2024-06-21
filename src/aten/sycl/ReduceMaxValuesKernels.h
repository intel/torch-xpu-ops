#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void max_values_kernel(TensorIterator& iter);

void max_kernel(TensorIterator& iter);

void max_all_kernel(TensorIterator& iter);

} // namespace at::native::xpu
