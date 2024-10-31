#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void shifted_chebyshev_polynomial_t_kernel(
    TensorIteratorBase& iterator);

TORCH_XPU_API void shifted_chebyshev_polynomial_u_kernel(
    TensorIteratorBase& iterator);

TORCH_XPU_API void shifted_chebyshev_polynomial_v_kernel(
    TensorIteratorBase& iterator);

TORCH_XPU_API void shifted_chebyshev_polynomial_w_kernel(
    TensorIteratorBase& iterator);

} // namespace at::native::xpu
