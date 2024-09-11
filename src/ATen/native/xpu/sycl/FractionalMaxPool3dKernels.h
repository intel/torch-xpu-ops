#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

void fractional_max_pool3d_out_kernel(
    Tensor& output,
    Tensor& indices,
    const Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const Tensor& randomSamples);

void fractional_max_pool3d_backward_out_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef pool_size /* unused */,
    IntArrayRef output_size,
    const Tensor& indices);

} // namespace at::native::xpu