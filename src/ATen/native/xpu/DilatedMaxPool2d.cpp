#include <ATen/core/Tensor.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xpu/sycl/DilatedMaxPool2d.h>
#include <comm/RegisterUtils.h>

#include <ATen/xpu/ops/max_pool2d_with_indices_backward_native.h>

namespace at {
<<<<<<< HEAD
namespace native {
TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_xpu)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& indices,
 const Tensor& gradInput) {
  xpu::max_pool2d_with_indices_backward_out_kernel(
      gradInput,
      gradOutput,
      input,
      indices,
=======

using namespace at::native;

void max_pool2d_with_indices_meta(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we
  // accept empty stride for this case
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(
        input.ndimension() == 4,
        "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(
      input,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output and indices */
  if (input.ndimension() == 3) {
    if (output.defined()) {
      at::xpu::resize_out(
          output,
          {nInputPlane, outputHeight, outputWidth},
          {},
          input.options().memory_format(memory_format));
    } else {
      output = at::xpu::create_out(
          {nInputPlane, outputHeight, outputWidth},
          {},
          input.options().memory_format(memory_format));
    }

    /* indices will contain the locations for each output point */
    if (indices.defined()) {
      at::xpu::resize_out(
          indices,
          {nInputPlane, outputHeight, outputWidth},
          {},
          input.options().memory_format(memory_format).dtype(kLong));
    } else {
      indices = at::xpu::create_out(
          {nInputPlane, outputHeight, outputWidth},
          {},
          input.options().memory_format(memory_format).dtype(kLong));
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

    /* indices will contain the locations for each output point */
    if (indices.defined()) {
      at::xpu::resize_out(
          indices,
          {nbatch, nInputPlane, outputHeight, outputWidth},
          {},
          input.options().memory_format(memory_format).dtype(kLong));
    } else {
      indices = at::xpu::create_out(
          {nbatch, nInputPlane, outputHeight, outputWidth},
          {},
          input.options().memory_format(memory_format).dtype(kLong));
    }
  }
}

Tensor& max_pool2d_with_indices_backward_meta(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    Tensor& gradInput) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      input.dtype() == gradOutput.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `gradOutput` but got dtype ",
      gradOutput.dtype());

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(
        input.ndimension() == 4,
        "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  /* XXX preserve the existing shape check behavior */
  const int64_t outputHeight_for_shape_check = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth_for_shape_check = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
      input,
      gradOutput,
      indices,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight_for_shape_check,
      outputWidth_for_shape_check,
      memory_format);

  auto options = input.options().memory_format(memory_format);
  if (gradInput.defined()) {
    at::xpu::resize_out(gradInput, input.sizes(), {}, options);
  } else {
    gradInput = at::xpu::create_out(input.sizes(), {}, options);
  }

  return gradInput;
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::max_pool2d_with_indices(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output;
  Tensor indices;
  max_pool2d_with_indices_meta(
      input,
>>>>>>> main
      kernel_size,
      stride,
      padding,
      dilation,
<<<<<<< HEAD
      ceil_mode);
}
} // namespace native
=======
      ceil_mode,
      output,
      indices);

  at::native::xpu::max_pool2d_with_indices_kernel(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::max_pool2d_with_indices_out(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& output,
    Tensor& indices) {
  max_pool2d_with_indices_meta(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  at::native::xpu::max_pool2d_with_indices_kernel(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  return std::tuple<Tensor&, Tensor&>(output, indices);
}

Tensor& XPUNativeFunctions::max_pool2d_with_indices_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices,
    Tensor& grad_input) {
  grad_input = max_pool2d_with_indices_backward_meta(
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices,
      grad_input);

  at::native::xpu::max_pool2d_with_indices_backward_kernel(
      grad_input,
      grad_output,
      self,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);

  return grad_input;
}

Tensor XPUNativeFunctions::max_pool2d_with_indices_backward(
    const Tensor& grad_output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  Tensor grad_input;
  max_pool2d_with_indices_backward_out(
      grad_output,
      self,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices,
      grad_input);

  return grad_input;
}

>>>>>>> main
} // namespace at
