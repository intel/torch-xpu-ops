#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DropoutKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

::std::tuple<Tensor, Tensor> XPUNativeFunctions::native_dropout(
    const Tensor& input,
    double p,
    ::std::optional<bool> train) {
  return at::native::xpu::dropout_kernel(input, p, train);
}

Tensor XPUNativeFunctions::native_dropout_backward(
    const Tensor& grad_output,
    const Tensor& mask,
    double scale) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::native_dropout_backward",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, mask, "xpu::native_dropout_backward", "mask");
  return at::native::xpu::dropout_backward_kernel(grad_output, mask, scale);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::_fused_dropout(
    const Tensor& self,
    double p,
    std::optional<at::Generator> generator) {
  return native::xpu::fused_dropout_kernel(self, p, generator);
}

} // namespace at
