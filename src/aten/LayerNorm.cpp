#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <aten/sycl/LayerNormKernels.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/library.h>

namespace at {

::std::tuple<at::Tensor, at::Tensor, at::Tensor> XPUNativeFunctions::
    native_layer_norm(
        const at::Tensor& input,
        at::IntArrayRef normalized_shape,
        const ::std::optional<at::Tensor>& weight,
        const ::std::optional<at::Tensor>& bias,
        double eps) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::native_layer_norm", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::native_layer_norm", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "xpu::native_layer_norm", "bias");
  return native::xpu::native_layer_norm(
      input, normalized_shape, weight, bias, eps);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor> XPUNativeFunctions::
    native_layer_norm_backward(
        const at::Tensor& grad_out,
        const at::Tensor& input,
        at::IntArrayRef normalized_shape,
        const at::Tensor& mean,
        const at::Tensor& rstd,
        const ::std::optional<at::Tensor>& weight,
        const ::std::optional<at::Tensor>& bias,
        ::std::array<bool, 3> output_mask) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, grad_out, "xpu::native_layer_norm_backward", "grad_out");
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::native_layer_norm_backward", "input");
  c10::impl::check_and_update_common_device(
      common_device, mean, "xpu::native_layer_norm_backward", "mean");
  c10::impl::check_and_update_common_device(
      common_device, rstd, "xpu::native_layer_norm_backward", "rstd");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::native_layer_norm_backward", "weight");
  c10::impl::check_and_update_common_device(
      common_device, bias, "xpu::native_layer_norm_backward", "bias");
  return native::xpu::native_layer_norm_backward(
      grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
}

} // namespace at
