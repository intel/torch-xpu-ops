#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor grid_sampler_2d_kernel(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners);

TORCH_XPU_API void grid_sampler_2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_grid,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask);

TORCH_XPU_API Tensor grid_sampler_3d_kernel(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners);

TORCH_XPU_API void grid_sampler_3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_grid,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask);

} // namespace at::native::xpu
