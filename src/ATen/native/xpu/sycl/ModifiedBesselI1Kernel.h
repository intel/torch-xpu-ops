#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void modified_bessel_i1_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
