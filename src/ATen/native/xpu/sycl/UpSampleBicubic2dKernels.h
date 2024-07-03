#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void upsample_bicubic2d_kernel(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w);

} // namespace at::native::xpu