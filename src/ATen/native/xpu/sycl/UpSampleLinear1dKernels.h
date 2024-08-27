#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {
void upsample_linear1d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales,
    Tensor& output);

void upsample_linear1d_backward_kernel(
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales,
    Tensor& grad_input);

} // namespace at::native::xpu