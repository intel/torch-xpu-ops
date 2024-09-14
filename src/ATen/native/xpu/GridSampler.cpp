#include <ATen/ATen.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/GridSamplerKernels.h>

namespace at {

Tensor XPUNativeFunctions::grid_sampler_2d(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  return native::xpu::grid_sampler_2d_kernel(
      input, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::grid_sampler_2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  native::xpu::grid_sampler_2d_backward_kernel(
      grad_input,
      grad_grid,
      grad_output,
      input,
      grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

Tensor XPUNativeFunctions::grid_sampler_3d(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  return native::xpu::grid_sampler_3d_kernel(
      input, grid, interpolation_mode, padding_mode, align_corners);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::grid_sampler_3d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  native::xpu::grid_sampler_3d_backward_kernel(
      grad_input,
      grad_grid,
      grad_output,
      input,
      grid,
      interpolation_mode,
      padding_mode,
      align_corners,
      output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

} // namespace at
