#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void cdist_kernel(
    Tensor& result,
    const Tensor& x1_expanded,
    const Tensor& x2_expanded,
    double p);

} // namespace at::native::xpu
