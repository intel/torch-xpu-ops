#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void adaptive_avg_pool2d_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input);

void adaptive_avg_pool2d_kernel(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size);

} // namespace at::native::xpu
