#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void linear_int4_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scale,
    const Tensor& weight_zero_point,
    const std::optional<Tensor>& weight_bias,
    Tensor& output);

} // namespace at::native::xpu
