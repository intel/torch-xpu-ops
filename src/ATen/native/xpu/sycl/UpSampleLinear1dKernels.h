#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void upsample_linear1d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales,
    const Tensor& output);

TORCH_XPU_API void upsample_linear1d_backward_kernel(
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales,
    const Tensor& grad_input);

} // namespace at::native::xpu
