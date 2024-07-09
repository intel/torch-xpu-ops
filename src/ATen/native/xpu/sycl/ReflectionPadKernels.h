#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

void reflection_pad2d_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef padding);

void reflection_pad2d_backward_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding);

} // namespace at::native::xpu