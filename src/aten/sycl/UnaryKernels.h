#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void abs_kernel(TensorIterator& iter);

} // at::native::xpu
