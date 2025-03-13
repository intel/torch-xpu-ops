#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/div_rtn.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/DepthwiseConv2dKernels.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conv_depthwise2d_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native {
Tensor& conv_depthwise2d_xpu_out(
    const Tensor& input_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& out) {
  TORCH_CHECK(kernel_size.size() == 2);
  TORCH_CHECK(stride.size() == 2);
  TORCH_CHECK(padding.size() == 2);
  TORCH_CHECK(dilation.size() == 2);

  auto input = input_.expect_contiguous();
  auto weight = weight_.expect_contiguous();
  auto bias = [&] {
    if (bias_opt.has_value() && bias_opt->defined()) {
      return bias_opt->expect_contiguous();
    }
    return c10::MaybeOwned<Tensor>::owned(std::in_place);
  }();

  xpu::conv_depthwise2d_forward_kernel(
      *input,
      out,
      *weight,
      *bias,
      kernel_size[1],
      kernel_size[0],
      stride[1],
      stride[0],
      padding[1],
      padding[0],
      dilation[1],
      dilation[0]);
  return out;
}

Tensor conv_depthwise2d_xpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const std::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  auto out = at::empty({0}, input.options());
  return conv_depthwise2d_xpu_out(
      input, weight, kernel_size, bias, stride, padding, dilation, out);
}

std::tuple<Tensor&, Tensor&> conv_depthwise2d_backward_xpu_out(
    const Tensor& grad_output_,
    const Tensor& self_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& grad_input,
    Tensor& grad_weight) {
  auto grad_output = grad_output_.expect_contiguous();

  if (grad_weight.defined()) {
    auto self = self_.expect_contiguous();
    xpu::conv_depthwise2d_grad_weight_kernel(
        *self,
        *grad_output,
        grad_weight,
        kernel_size[1],
        kernel_size[0],
        stride[1],
        stride[0],
        padding[1],
        padding[0],
        dilation[1],
        dilation[0]);
  }

  if (grad_input.defined()) {
    auto weight = weight_.expect_contiguous();
    xpu::conv_depthwise2d_backward_kernel(
        self_,
        *grad_output,
        grad_input,
        *weight,
        kernel_size[1],
        kernel_size[0],
        stride[1],
        stride[0],
        padding[1],
        padding[0],
        dilation[1],
        dilation[0]);
  }
  return std::forward_as_tuple(grad_input, grad_weight);
}

std::tuple<Tensor, Tensor> conv_depthwise2d_backward_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    std::array<bool, 2> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  }
  return conv_depthwise2d_backward_xpu_out(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      dilation,
      grad_input,
      grad_weight);
}

REGISTER_XPU_DISPATCH(
    conv_depthwise2d_backward_stub,
    &conv_depthwise2d_backward_xpu);

} // namespace at::native