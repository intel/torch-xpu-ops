#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor& max_unpooling2d_forward_kernel(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size);

TORCH_XPU_API Tensor& max_unpooling3d_forward_kernel(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding);

} // namespace at::native::xpu
