#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {
void upsample_nearest2d_out_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

void upsample_nearest2d_backward_out_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);
} // namespace at::native::xpu