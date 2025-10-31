#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void linalg_qr_kernel(
    const Tensor& A,
    std::string_view mode,
    const Tensor& Q,
    const Tensor& R);

} // namespace at::native::xpu
