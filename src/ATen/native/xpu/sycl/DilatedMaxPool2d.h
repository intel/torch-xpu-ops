#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void max_pool2d_with_indices_kernel(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& output,
    const Tensor& indices);

TORCH_XPU_API void max_pool2d_with_indices_backward_kernel(
    const Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode);

} // namespace at::native::xpu
