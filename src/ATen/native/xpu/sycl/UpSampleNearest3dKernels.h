#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void upsample_nearest3d_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool is_exact);

TORCH_XPU_API void upsample_nearest3d_backward_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    bool is_exact);

} // namespace at::native::xpu
