#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void avg_pool3d_kernel(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    const Tensor& output);

TORCH_XPU_API void avg_pool3d_backward_kernel(
    const Tensor& gradOutput_,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    const Tensor& gradInput);

} // namespace at::native::xpu
