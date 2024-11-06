#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void scaled_modified_bessel_k0_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu