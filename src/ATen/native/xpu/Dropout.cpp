#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Dropout.h>
#include <ATen/native/xpu/sycl/DropoutKernels.h>

#include <xpu/ATen/ops/native_dropout_backward_native.h>
#include <xpu/ATen/ops/native_dropout_native.h>

#include <ATen/native/xpu/sycl/Dropout.h>
#include <comm/xpu_aten.h>

namespace at {

namespace native {
::std::tuple<Tensor, Tensor> native_dropout_xpu(
    const Tensor& input,
    double p,
    ::std::optional<bool> train) {
  return at::native::xpu::dropout_kernel(input, p, train);
}

Tensor native_dropout_backward_xpu(
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

} // namespace native
} // namespace at
