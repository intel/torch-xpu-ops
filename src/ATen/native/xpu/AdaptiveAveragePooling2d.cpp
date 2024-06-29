#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/AdaptiveAveragePooling2dKernels.h>

namespace at {

Tensor XPUNativeFunctions::_adaptive_avg_pool2d_backward(
    const Tensor& grad_output_,
    const Tensor& input_) {
  globalContext().alertNotDeterministic("_adaptive_avg_pool2d_backward");
  Tensor grad_input;
  if (input_.numel() != 0) {
    Tensor input, grad_output;
    if (input_.ndimension() == 3) {
      input = input_.contiguous();
      grad_output = grad_output_.contiguous();
      grad_input = at::empty_like(input);
    } else {
      auto smf = input_.suggest_memory_format();
      input = input_.contiguous(smf);
      grad_output = grad_output_.contiguous(smf);
      grad_input = at::empty_like(input_, smf);
    }
    native::xpu::adaptive_avg_pool2d_backward_out_kernel(
        grad_input, grad_output, input);
  } else {
    grad_input = at::zeros_like(input_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  return grad_input;
}

Tensor& XPUNativeFunctions::adaptive_avg_pool2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  native::xpu::adaptive_avg_pool2d_out_kernel(output, input, output_size);
  return output;
}

Tensor XPUNativeFunctions::_adaptive_avg_pool2d(
    at::Tensor const& input,
    IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  native::xpu::adaptive_avg_pool2d_out_kernel(output, input, output_size);
  return output;
}

} // namespace at
