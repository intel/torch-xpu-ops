
#include <ATen/core/op_registration/adaption.h>
#include <ATen/div_rtn.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <ATen/native/xpu/sycl/LinearInt4.h>
#include <comm/xpu_aten.h>

namespace at::native {
Tensor& linear_int4_xpu(
    const Tensor& input,
    const Tensor& weight,
    int qGroupSize,
    const Tensor& weight_scale_zero_point) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::linear_int4", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::linear_int4", "weight");
  c10::impl::check_and_update_common_device(
      common_device,
      weight_scale_zero_point,
      "xpu::linear_int4",
      "weight_scale_zero_point");
  Tensor output = at::empty({0}, input.options());

  at::native::xpu::linear_int4_kernel(
      input, weight, qGroupSize, weight_scale_zero_point, output);
  return output;
}
} // namespace at::native
