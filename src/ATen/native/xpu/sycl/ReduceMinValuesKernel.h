#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void min_values_kernel(TensorIterator& iter);

void min_launch_kernel(TensorIterator& iter);

void min_all_launch_kernel(TensorIterator& iter);

} // namespace at::native::xpu
