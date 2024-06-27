#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/ReflectionPadKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

Tensor& XPUNativeFunctions::reflection_pad2d_out(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  native::xpu::reflection_pad2d_kernel(output, input, padding);
  return output;
}

Tensor XPUNativeFunctions::reflection_pad2d(
    const Tensor& input,
    IntArrayRef padding) {
  auto output = at::empty({0}, input.options());
  native::xpu::reflection_pad2d_kernel(output, input, padding);
  return output;
}

Tensor& XPUNativeFunctions::reflection_pad2d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("reflection_pad2d_backward_out_xpu");
  grad_input.resize_as_(input);
  grad_input.zero_();
  native::xpu::reflection_pad2d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor XPUNativeFunctions::reflection_pad2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("reflection_pad2d_backward_xpu");
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  native::xpu::reflection_pad2d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

} // namespace at