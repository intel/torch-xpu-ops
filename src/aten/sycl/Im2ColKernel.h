#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void im2col_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride);

} // namespace at::native::xpu
