#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void max_pool2d_with_indices_kernel(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices);

Tensor& max_pool2d_with_indices_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);

} // namespace at::native::xpu
