#pragma once

#include <ATen/Tensor.h>

namespace at::native::xpu {

void adaptive_max_pool3d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output,
    Tensor& indices);

void adaptive_max_pool3d_backward_kernel(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    Tensor& gradInput);

} // namespace at::native::xpu
