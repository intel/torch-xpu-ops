#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Padding.h>
#include <ATen/native/xpu/sycl/ReflectionPadKernels.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#include <xpu/ATen/ops/reflection_pad1d_backward_native.h>
#include <xpu/ATen/ops/reflection_pad1d_native.h>
#include <xpu/ATen/ops/reflection_pad2d_backward_native.h>
#include <xpu/ATen/ops/reflection_pad2d_native.h>
#include <xpu/ATen/ops/reflection_pad3d_backward_native.h>
#include <xpu/ATen/ops/reflection_pad3d_native.h>
#include <ATen/TensorMeta.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(reflection_pad1d_out_xpu)
(const Tensor& input_, IntArrayRef padding, const Tensor& output) {
  xpu::reflection_pad1d_kernel(output, input_, padding);
}

TORCH_IMPL_FUNC(reflection_pad1d_backward_out_xpu)
(const Tensor& grad_output,
 const Tensor& input,
 IntArrayRef padding,
 const Tensor& grad_input) {
  xpu::reflection_pad1d_backward_kernel(
      grad_input, grad_output, input, padding);
}

Tensor& reflection_pad2d_out_xpu(
    const Tensor& input,
    IntArrayRef padding,
    Tensor& output) {
  native::xpu::reflection_pad2d_kernel(output, input, padding);
  return output;
}

Tensor reflection_pad2d_xpu(const Tensor& input, IntArrayRef padding) {
  auto output = at::empty({0}, input.options());
  native::xpu::reflection_pad2d_kernel(output, input, padding);
  return output;
}

Tensor& reflection_pad2d_backward_out_xpu(
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

Tensor reflection_pad2d_backward_xpu(
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

TORCH_IMPL_FUNC(reflection_pad3d_out_xpu)
(const Tensor& input_, IntArrayRef padding, const Tensor& output) {
  xpu::reflection_pad3d_kernel(output, input_, padding);
}

TORCH_IMPL_FUNC(reflection_pad3d_backward_out_xpu)
(const Tensor& grad_output,
 const Tensor& input,
 IntArrayRef padding,
 const Tensor& grad_input) {
  xpu::reflection_pad3d_backward_kernel(
      grad_input, grad_output, input, padding);
}

} // namespace native
} // namespace at