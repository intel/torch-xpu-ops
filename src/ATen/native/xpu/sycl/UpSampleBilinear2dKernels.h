#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void upsample_bilinear2d_out_kernel(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w);

void upsample_bilinear2d_backward_out_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w);

} // namespace at::native::xpu