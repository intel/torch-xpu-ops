#pragma once

#include <ATen/Tensor.h>

namespace at::native::xpu {

void adaptive_max_pool2d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output,
    Tensor& indices);

void adaptive_max_pool2d_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    Tensor& grad_input);

} // namespace at::native::xpu
