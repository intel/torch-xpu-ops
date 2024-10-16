#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void upsample_bicubic2d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w);

void upsample_bicubic2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w);
    
} // namespace at::native::xpu
