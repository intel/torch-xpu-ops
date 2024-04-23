#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>

namespace at {

Tensor XPUNativeFunctions::_softmax(
    const Tensor& self,
    int64_t dim,
    bool half_to_float) {
  Tensor self_cpu = self.to(Device(kCPU));
  Tensor out = at::_softmax(self_cpu, dim, half_to_float);
  return out.to(Device(kXPU));
}

Tensor& XPUNativeFunctions::_softmax_out(
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  Tensor out_cpu = out.to(Device(kCPU));
  Tensor self_cpu = self.to(Device(kCPU));
  at::_softmax_out(out_cpu, self_cpu, dim, half_to_float);
  out.copy_(out_cpu);
  return out;
}

Tensor XPUNativeFunctions::_softmax_backward_data(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  Tensor grad_output_cpu = grad_output.to(Device(kCPU));
  Tensor output_cpu = output.to(Device(kCPU));
  Tensor grad_input =
      at::_softmax_backward_data(grad_output_cpu, output_cpu, dim, input_dtype);
  return grad_input.to(Device(kXPU));
}

Tensor& XPUNativeFunctions::_softmax_backward_data_out(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    Tensor& grad_input) {
  Tensor grad_output_cpu = grad_output.to(Device(kCPU));
  Tensor output_cpu = output.to(Device(kCPU));
  Tensor grad_input_cpu = grad_input.to(Device(kCPU));
  at::_softmax_backward_data_out(
      grad_input_cpu, grad_output_cpu, output_cpu, dim, input_dtype);
  grad_input.copy_(grad_input_cpu);
  return grad_input;
}

} // namespace at