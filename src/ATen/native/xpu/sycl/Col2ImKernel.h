#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void col2im_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride);

} // namespace at::native::xpu
