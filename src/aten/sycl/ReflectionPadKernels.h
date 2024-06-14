#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void reflection_pad2d_out_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef padding);

void reflection_pad2d_backward_out_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding);

} // namespace at::native::xpu