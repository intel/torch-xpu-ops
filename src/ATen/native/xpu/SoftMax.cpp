#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/SoftMaxKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {

Tensor& _softmax_meta(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    Tensor& out) {
  int64_t dim_ = maybe_wrap_dim(dim, input.dim());

  auto output_options =
      input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  if (half_to_float) {
    output_options = output_options.dtype(ScalarType::Float);
  }

  int64_t input_dim = input.dim() > 0 ? input.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_dim,
      "dim must be non-negative and less than input dimensions");

  if (out.defined()) {
    xpu::resize_out(out, input.sizes(), {}, output_options);
  } else {
    out = xpu::create_out(input.sizes(), {}, output_options);
  }

  return out;
}

Tensor XPUNativeFunctions::_softmax(
    const Tensor& self,
    int64_t dim,
    bool half_to_float) {
  Tensor out;
  out = _softmax_meta(self, dim, half_to_float, out);
  native::xpu::_softmax_kernel(self, dim, half_to_float, out);
  return out;
}

Tensor& XPUNativeFunctions::_softmax_out(
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::_softmax_out_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::_softmax_out_out", "self");
  out = _softmax_meta(self, dim, half_to_float, out);
  return native::xpu::_softmax_kernel(self, dim, half_to_float, out);
}

Tensor XPUNativeFunctions::_log_softmax(
    const Tensor& self,
    int64_t dim,
    bool half_to_float) {
  Tensor out;
  out = _softmax_meta(self, dim, half_to_float, out);
  native::xpu::_log_softmax_kernel(self, dim, half_to_float, out);
  return out;
}

Tensor& XPUNativeFunctions::_log_softmax_out(
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::_log_softmax_out_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::_log_softmax_out_out", "self");
  out = _softmax_meta(self, dim, half_to_float, out);
  return native::xpu::_log_softmax_kernel(self, dim, half_to_float, out);
}

Tensor& _softmax_backward_data_meta(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    Tensor& grad_input) {
  TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
  checkSameSize("softmax_backward", grad_arg, output_arg);

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());

  auto grad_input_options =
      grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  bool half_to_float = grad.scalar_type() != input_dtype;
  if (half_to_float) {
    if (grad.scalar_type() == ScalarType::Float &&
        input_dtype == ScalarType::Half) {
      grad_input_options = grad_input_options.dtype(ScalarType::Half);
    }
  }

  int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < grad_dim,
      "dim must be non-negative and less than input dimensions");

  if (grad_input.defined()) {
    xpu::resize_out(grad_input, grad.sizes(), {}, grad_input_options);
  } else {
    grad_input = xpu::create_out(grad.sizes(), {}, grad_input_options);
  }

  return grad_input;
}

Tensor XPUNativeFunctions::_softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "xpu::_softmax_backward_data", "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::_softmax_backward_data", "output");
  Tensor grad_input;
  grad_input = _softmax_backward_data_meta(
      grad_output, output, dim, input_dtype, grad_input);
  native::xpu::_softmax_backward_kernel(
      grad_output, output, dim, false, grad_input);
  return grad_input;
}

Tensor& XPUNativeFunctions::_softmax_backward_data_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype,
    Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::_softmax_backward_data_out_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::_softmax_backward_data_out_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::_softmax_backward_data_out_out", "output");
  grad_input = _softmax_backward_data_meta(
      grad_output, output, dim, input_dtype, grad_input);
  return native::xpu::_softmax_backward_kernel(
      grad_output, output, dim, false, grad_input);
}

Tensor XPUNativeFunctions::_log_softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::_log_softmax_backward_data",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::_log_softmax_backward_data", "output");
  Tensor grad_input;
  grad_input = _softmax_backward_data_meta(
      grad_output, output, dim, input_dtype, grad_input);
  native::xpu::_log_softmax_backward_kernel(
      grad_output, output, dim, false, grad_input);
  return grad_input;
}

Tensor& XPUNativeFunctions::_log_softmax_backward_data_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype,
    Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::_log_softmax_backward_data_out_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::_log_softmax_backward_data_out_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device,
      output,
      "xpu::_log_softmax_backward_data_out_out",
      "output");
  grad_input = _softmax_backward_data_meta(
      grad_output, output, dim, input_dtype, grad_input);
  return native::xpu::_log_softmax_backward_kernel(
      grad_output, output, dim, false, grad_input);
}

} // namespace at
