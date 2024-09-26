#pragma once

#include <ATen/Tensor.h>

namespace at::native::xpu {

void adaptive_avg_pool3d_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input);

void adaptive_avg_pool3d_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef& output_size);

} // namespace at::native::xpu