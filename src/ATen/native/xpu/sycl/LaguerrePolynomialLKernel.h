#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void laguerre_polynomial_l_kernel(TensorIteratorBase& iterator);

} // namespace at::native::xpu
