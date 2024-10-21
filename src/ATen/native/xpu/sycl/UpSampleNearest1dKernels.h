#pragma once

#include <ATen/native/xpu/UpSample.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void upsample_nearest1d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales,
    bool is_exact);

TORCH_XPU_API void upsample_nearest1d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    bool is_exact);

} // namespace at::native::xpu
