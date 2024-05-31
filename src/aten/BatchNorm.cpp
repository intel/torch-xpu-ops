#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <aten/sycl/BatchNormKernels.h>

namespace at {

::std::tuple<Tensor, Tensor> XPUNativeFunctions::batch_norm_stats(
    const Tensor& input,
    double eps) {
  return native::xpu::batch_norm_stats_kernel(input, eps);
}

Tensor XPUNativeFunctions::batch_norm_elemt(
    const Tensor& input,
    const ::std::optional<Tensor>& weight,
    const ::std::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::batch_norm_elemt", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::batch_norm_elemt", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "xpu::batch_norm_elemt", "bias");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::batch_norm_elemt", "mean");
  c10::impl::check_and_update_common_device(
      common_device, invstd, "xpu::batch_norm_elemt", "invstd");
  auto output = at::empty_like(input);
  native::xpu::batch_norm_elemt_kernel(
      output, input, weight, bias, mean, invstd);
  return output;
}

Tensor& XPUNativeFunctions::batch_norm_elemt_out(
    const Tensor& input,
    const ::std::optional<Tensor>& weight,
    const ::std::optional<Tensor>& bias,
    const Tensor& mean,
    const Tensor& invstd,
    double eps,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::batch_norm_elemt_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::batch_norm_elemt_out", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::batch_norm_elemt_out", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "xpu::batch_norm_elemt_out", "bias");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::batch_norm_elemt_out", "mean");
  c10::impl::check_and_update_common_device(
      common_device, invstd, "xpu::batch_norm_elemt_out", "invstd");
  native::xpu::batch_norm_elemt_kernel(out, input, weight, bias, mean, invstd);
  return out;
}

} // namespace at
