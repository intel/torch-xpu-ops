#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>

namespace at {

// Following three functions are currently not use by sdpa
// TODO: Do we need to register them as well?
// Tensor softmax(const Tensor& input_, const int64_t dim_, c10::optional<ScalarType> dtype) {
//   Tensor output_ = softmax(input_.to(Device(kCPU)), dim_, dtype);
//   return output_.to(Device(kXPU));
// }

// Tensor& softmax_out(
//     const Tensor& input_,
//     const int64_t dim_,
//     c10::optional<ScalarType> dtype,
//     Tensor& output_) {
//   Tensor output_cpu = output_.to(Device(kCPU));
//   softmax_out(input_.to(Device(kCPU)), dim_, dtype, output_cpu);
//   output_.copy_(output_cpu);
//   return output_;
// }

// Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float) {
//   Tensor out = _softmax(self.to(Device(kCPU)), dim, half_to_float);
//   return out.to(Device(kXPU));
// }

Tensor & XPUNativeFunctions::_softmax_out(const Tensor & self, int64_t dim, bool half_to_float, Tensor & out) {
  Tensor out_cpu = out.to(Device(kCPU));
  _softmax_out(self.to(Device(kCPU)), dim, half_to_float, out_cpu);
  out.copy_(out_cpu);
  return out;
}

// TODO: This one too
// Tensor XPUNativeFunctions::_softmax_backward_data(
//     const Tensor & grad_output,
//     const Tensor & output,
//     int64_t dim,
//     ScalarType input_dtype) {
//   Tensor grad_input = _softmax_backward_data(
//       grad_output.to(Device(kCPU)),
//       output.to(Device(kCPU)),
//       dim,
//       input_dtype);
//   return grad_input.to(Device(kCPU));
// }

Tensor & XPUNativeFunctions::_softmax_backward_data_out(
    const Tensor & grad_output,
    const Tensor & output, 
    nt64_t dim,
    ScalarType input_dtype,
    Tensor & grad_input) {
  Tensor grad_input_cpu = grad_input.to(Device(kCPU));
  _softmax_backward_data_out(
      grad_output.to(Device(kCPU)), 
      output.to(Device(kCPU)),
      dim,
      input_dtype,
      grad_input_cpu);
  grad_input.copy_(grad_input_cpu);
  return grad_input;
}

} // namespace at