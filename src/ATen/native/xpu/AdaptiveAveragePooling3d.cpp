#include <ATen/ATen.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xpu/sycl/AdaptiveAveragePooling3dKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

Tensor& XPUNativeFunctions::adaptive_avg_pool3d_out(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  at::native::xpu::adaptive_avg_pool3d_kernel(output, input, output_size);
  return output;
}

Tensor XPUNativeFunctions::_adaptive_avg_pool3d(
    const Tensor& input,
    IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool3d_out(input, output_size, output);
  return output;
}

Tensor& XPUNativeFunctions::adaptive_avg_pool3d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    Tensor& grad_input) {
  globalContext().alertNotDeterministic("adaptive_avg_pool3d_backward_xpu");
  at::native::xpu::adaptive_avg_pool3d_backward_kernel(
      grad_input, grad_output, input);
  return grad_input;
}

Tensor XPUNativeFunctions::_adaptive_avg_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& input) {
  globalContext().alertNotDeterministic("adaptive_avg_pool3d_backward_xpu");
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::xpu::adaptive_avg_pool3d_backward_kernel(
      grad_input, grad_output, input);
  return grad_input;
}

} // namespace at
