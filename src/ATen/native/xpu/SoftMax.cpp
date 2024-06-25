
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/SoftMaxKernels.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/xpu/ops/_log_softmax_backward_data_native.h>
#include <ATen/xpu/ops/_softmax_backward_data_native.h>
#include <ATen/xpu/ops/_softmax_native.h>
namespace at::native {

TORCH_IMPL_FUNC(softmax_xpu_out)
(const Tensor& input,
 const int64_t dim,
 const bool half_to_float,
 const Tensor& output) {
  xpu::_softmax_kernel(input, dim, half_to_float, output);
}

TORCH_IMPL_FUNC(softmax_backward_xpu_out)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype,
 const Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::_softmax_backward_data_out_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      output,
      "xpu::_softmax_backward_data_out_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::_softmax_backward_data_out_out", "output");

  native::xpu::_softmax_backward_kernel(grad, output, dim, false, grad);
}

TORCH_IMPL_FUNC(log_softmax_backward_xpu_out)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype,
 const Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::_log_softmax_backward_data_out_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      output,
      "xpu::_log_softmax_backward_data_out_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device,
      output,
      "xpu::_log_softmax_backward_data_out_out",
      "output");
  native::xpu::_log_softmax_backward_kernel(
      grad, output, dim, false, grad_input);
}
} // namespace at::native
