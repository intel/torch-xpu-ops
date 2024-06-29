#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Pool.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/AveragePool2dKernels.h>
#include <comm/RegisterUtils.h>

namespace at {
using namespace at::native;
using namespace at::native::xpu;

Tensor& avg_pool2d_meta(
    const Tensor& input,
    Tensor& output,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  const int64_t dH = stride.empty() ? kH : stride[0];
  const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  const int64_t padH = padding[0];
  const int64_t padW = padding.size() == 1 ? padH : padding[1];

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  pool2d_shape_check(
      input,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      1,
      1,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output */
  if (input.ndimension() == 3) {
    if (output.defined()) {
      at::xpu::resize_out(
          output,
          {nInputPlane, outputHeight, outputWidth},
          {},
          input.options());
    } else {
      output = at::xpu::create_out(
          {nInputPlane, outputHeight, outputWidth}, {}, input.options());
    }
  } else {
    if (output.defined()) {
      at::xpu::resize_out(
          output,
          {nbatch, nInputPlane, outputHeight, outputWidth},
          {},
          input.options().memory_format(memory_format));
    } else {
      output = at::xpu::create_out(
          {nbatch, nInputPlane, outputHeight, outputWidth},
          {},
          input.options().memory_format(memory_format));
    }
  }

  return output;
}

Tensor& avg_pool2d_backward_meta(
    const Tensor& gradOutput_,
    Tensor& grad_input,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of "
      "two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3); // number of channels (or colors)
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  avg_pool2d_backward_shape_check(
      input,
      gradOutput_,
      nbatch,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  if (grad_input.defined()) {
    at::xpu::resize_out(
        grad_input,
        input.sizes(),
        {},
        input.options().memory_format(memory_format));
  } else {
    grad_input = at::xpu::create_out(
        input.sizes(), {}, input.options().memory_format(memory_format));
  }
  return grad_input;
}

Tensor XPUNativeFunctions::avg_pool2d(
    const Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output;
  output = avg_pool2d_meta(
      input,
      output,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  at::native::xpu::avg_pool2d_kernel(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      output);
  return output;
}

Tensor& XPUNativeFunctions::avg_pool2d_out(
    const Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  avg_pool2d_meta(
      input,
      output,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  at::native::xpu::avg_pool2d_kernel(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      output);
  return output;
}

Tensor XPUNativeFunctions::avg_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor grad_input;
  grad_input = avg_pool2d_backward_meta(
      grad_output,
      grad_input,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  at::native::xpu::avg_pool2d_backward_kernel(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      grad_input);
  return grad_input;
}

Tensor& XPUNativeFunctions::avg_pool2d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_input) {
  avg_pool2d_backward_meta(
      grad_output,
      grad_input,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  at::native::xpu::avg_pool2d_backward_kernel(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      grad_input);
  return grad_input;
}

} // namespace at
