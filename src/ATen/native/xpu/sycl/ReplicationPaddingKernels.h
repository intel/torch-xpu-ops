#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void replication_pad1d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad1d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad2d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad3d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding);

} // namespace at::native::xpu
