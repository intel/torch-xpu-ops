#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Pool.h>
#include <ATen/native/xpu/sycl/AveragePool3dKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {
using namespace at::native;
using namespace at::native::xpu;

Tensor& avg_pool3d_meta(
    const Tensor& input,
    Tensor& output,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.size(0);
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t otime =
      pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  pool3d_shape_check(
      input,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      1,
      1,
      1,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth,
      "avg_pool3d()",
      /*check_input_size=*/true);

  /* resize output */
  if (input.ndimension() == 4) {
    if (output.defined()) {
      at::xpu::resize_out(
          output, {nslices, otime, oheight, owidth}, {}, input.options());
    } else {
      output = at::xpu::create_out(
          {nslices, otime, oheight, owidth}, {}, input.options());
    }
  } else {
    if (output.defined()) {
      at::xpu::resize_out(
          output,
          {nbatch, nslices, otime, oheight, owidth},
          {},
          input.options());
    } else {
      output = at::xpu::create_out(
          {nbatch, nslices, otime, oheight, owidth}, {}, input.options());
    }
  }

  return output;
}

Tensor& avg_pool3d_backward_meta(
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
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  /* XXX shape check behavior from TH */
  const int64_t otime_for_shape_check =
      pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight_for_shape_check =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_shape_check =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  avg_pool3d_backward_shape_check(
      input,
      gradOutput_,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      itime,
      iheight,
      iwidth,
      otime_for_shape_check,
      oheight_for_shape_check,
      owidth_for_shape_check,
      "avg_pool3d_backward()");

  if (grad_input.defined()) {
    at::xpu::resize_out(grad_input, input.sizes(), {}, input.options());
  } else {
    grad_input = at::xpu::create_out(input.sizes(), {}, input.options());
  }
  return grad_input;
}

Tensor XPUNativeFunctions::avg_pool3d(
    const Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output;
  output = avg_pool3d_meta(
      input,
      output,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  at::native::xpu::avg_pool3d_kernel(
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

Tensor& XPUNativeFunctions::avg_pool3d_out(
    const Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& output) {
  avg_pool3d_meta(
      input,
      output,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);

  at::native::xpu::avg_pool3d_kernel(
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

Tensor XPUNativeFunctions::avg_pool3d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor grad_input;
  grad_input = avg_pool3d_backward_meta(
      grad_output,
      grad_input,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  at::native::xpu::avg_pool3d_backward_kernel(
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

Tensor& XPUNativeFunctions::avg_pool3d_backward_out(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    Tensor& grad_input) {
  avg_pool3d_backward_meta(
      grad_output,
      grad_input,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
  at::native::xpu::avg_pool3d_backward_kernel(
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
