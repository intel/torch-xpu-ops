#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void avg_pool2d_kernel(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output);

void avg_pool2d_backward_kernel(
    const Tensor& gradOutput_,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& gradInput);

} // namespace at::native::xpu
