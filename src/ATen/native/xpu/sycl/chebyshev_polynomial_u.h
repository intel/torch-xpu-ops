#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator);

} // namespace at::native::xpu