#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void reflection_pad1d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef padding);

TORCH_XPU_API void reflection_pad1d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void reflection_pad2d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef padding);

TORCH_XPU_API void reflection_pad2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void reflection_pad3d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef padding);

TORCH_XPU_API void reflection_pad3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding);

} // namespace at::native::xpu
