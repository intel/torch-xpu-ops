#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <comm/RegisterUtils.h>

namespace at {

void meta_func__softmax(
    Tensor& out,
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float) {
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
    at::xpu::resize_out(out, input.sizes(), {}, output_options);
  } else {
    out = at::xpu::create_out(input.sizes(), {}, output_options);
  }
}

Tensor& XPUNativeFunctions::_softmax_out(
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::_softmax_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::_softmax_out", "self");
  meta_func__softmax(out, self, dim, half_to_float);
  Tensor out_cpu = out.to(Device(kCPU));
  Tensor self_cpu = self.to(Device(kCPU));
  at::_softmax_out(out_cpu, self_cpu, dim, half_to_float);
  out.copy_(out_cpu);
  return out;
}

Tensor XPUNativeFunctions::_softmax(
    const Tensor& self,
    int64_t dim,
    bool half_to_float) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::_softmax_out", "self");
  Tensor out;
  meta_func__softmax(out, self, dim, half_to_float);
  Tensor out_cpu = out.to(Device(kCPU));
  Tensor self_cpu = self.to(Device(kCPU));
  at::_softmax_out(out_cpu, self_cpu, dim, half_to_float);
  out.copy_(out_cpu);
  return out;
}

void meta_func__softmax_backward_data(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    Tensor& grad_input) {
  TensorArg grad_output_arg{grad_output, "grad_output", 1},
      output_arg{output, "output", 2};
  checkSameSize("softmax_backward", grad_output_arg, output_arg);

  int64_t dim_ = maybe_wrap_dim(dim, grad_output.dim());

  auto grad_input_options =
      grad_output.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  bool half_to_float = grad_output.scalar_type() != input_dtype;
  if (half_to_float) {
    if (grad_output.scalar_type() == ScalarType::Float &&
        input_dtype == ScalarType::Half) {
      grad_input_options = grad_input_options.dtype(ScalarType::Half);
    }
  }

  int64_t grad_dim = grad_output.dim() > 0 ? grad_output.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < grad_dim,
      "dim must be non-negative and less than input dimensions");
  if (grad_input.defined()) {
    at::xpu::resize_out(
        grad_input, grad_output.sizes(), {}, grad_input_options);
  } else {
    grad_input =
        at::xpu::create_out(grad_output.sizes(), {}, grad_input_options);
  }
}

Tensor& XPUNativeFunctions::_softmax_backward_data_out(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    Tensor& grad_input) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "xpu::_softmax_backward_data_out",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::_softmax_backward_data_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::_softmax_backward_data_out", "output");
  meta_func__softmax_backward_data(
      grad_output, output, dim, input_dtype, grad_input);
  Tensor grad_output_cpu = grad_output.to(Device(kCPU));
  Tensor output_cpu = output.to(Device(kCPU));
  Tensor grad_input_cpu = grad_input.to(Device(kCPU));
  at::_softmax_backward_data_out(
      grad_input_cpu, grad_output_cpu, output_cpu, dim, input_dtype);
  grad_input.copy_(grad_input_cpu);
  return grad_input;
}

Tensor XPUNativeFunctions::_softmax_backward_data(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "xpu::_softmax_backward_data_out",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, output, "xpu::_softmax_backward_data_out", "output");
  Tensor grad_input;
  meta_func__softmax_backward_data(
      grad_output, output, dim, input_dtype, grad_input);
  Tensor grad_output_cpu = grad_output.to(Device(kCPU));
  Tensor output_cpu = output.to(Device(kCPU));
  Tensor grad_input_cpu = grad_input.to(Device(kCPU));
  at::_softmax_backward_data_out(
      grad_input_cpu, grad_output_cpu, output_cpu, dim, input_dtype);
  grad_input.copy_(grad_input_cpu);
  return grad_input;
}

} // namespace at