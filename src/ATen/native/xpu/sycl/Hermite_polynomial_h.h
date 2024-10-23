#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void hermite_polynomial_h_kernel(TensorIteratorBase& iterator);

} // namespace at::native::xpu