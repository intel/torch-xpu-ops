
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/AdaptiveAveragePooling2dKernel.h>
#include <comm/xpu_aten.h>

namespace at {

Tensor XPUNativeFunctions::_adaptive_avg_pool2d_backward(
    const Tensor& grad_output_,
    const Tensor& input_) {
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

} // namespace at
