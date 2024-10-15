#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void adaptive_avg_pool3d_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input);

TORCH_XPU_API void adaptive_avg_pool3d_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef& output_size);

} // namespace at::native::xpu