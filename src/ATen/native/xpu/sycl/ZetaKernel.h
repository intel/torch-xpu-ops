#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void zeta_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu