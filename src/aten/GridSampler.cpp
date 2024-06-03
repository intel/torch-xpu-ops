#include <ATen/ATen.h>
#include <ATen/core/op_registration/adaption.h>

#include <ATen/XPUNativeFunctions.h>
#include <aten/sycl/GridSamplerKernels.h>

namespace at {

Tensor XPUNativeFunctions::grid_sampler_2d(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::grid_sampler_2d", "input");

  c10::impl::check_and_update_common_device(
      common_device, grid, "xpu::grid_sampler_2d", "grid");
  return native::xpu::grid_sampler_2d_kernel(
      input, grid, interpolation_mode, padding_mode, align_corners);
}

} // namespace at