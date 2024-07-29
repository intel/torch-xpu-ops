#include <ATen/ATen.h>
#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/UpSampleLinear1dKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>
#include "ATen/core/ATen_fwd.h"

namespace at {

void upsample_linear1d_meta(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales,
    Tensor& output) {
  auto full_output_size =
      at::native::xpu::upsample_1d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  if (output.defined()) {
    at::xpu::resize_out(output, full_output_size, {}, input.options());
  } else {
    output = at::xpu::create_out(full_output_size, {}, input.options());
  }
}
void upsample_linear1d_backward_meta(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales,
    Tensor& grad_input) {
  auto full_output_size =
      at::native::xpu::upsample_1d_common_check(input_size, output_size);

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  check_dim_size(grad_output, 3, 0, full_output_size[0]);
  check_dim_size(grad_output, 3, 1, full_output_size[1]);
  check_dim_size(grad_output, 3, 2, full_output_size[2]);

  if (grad_input.defined()) {
    at::xpu::resize_out(grad_input, input_size, {}, grad_output.options());
  } else {
    grad_input = at::xpu::create_out(input_size, {}, grad_output.options());
  }
}

Tensor XPUNativeFunctions::upsample_linear1d(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales) {
  Tensor output;
  return upsample_linear1d_out(
      input, output_size, align_corners, scales, output);
}

Tensor& XPUNativeFunctions::upsample_linear1d_out(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales,
    Tensor& output) {
  upsample_linear1d_meta(input, output_size, align_corners, scales, output);

  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  native::xpu::upsample_linear1d_kernel(
      input, output_size, align_corners, scales, output);
  return output;
}
Tensor XPUNativeFunctions::upsample_linear1d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales) {
  Tensor grad_input;
  return upsample_linear1d_backward_out(
      grad_output, output_size, input_size, align_corners, scales, grad_input);
}

Tensor& XPUNativeFunctions::upsample_linear1d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales,
    Tensor& grad_input) {
  upsample_linear1d_backward_meta(
      grad_output, output_size, input_size, align_corners, scales, grad_input);

  TensorArg grad_output_arg{grad_output, "grad_output", 1},
      grad_input_arg{grad_input, "grad_input", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});
  native::xpu::upsample_linear1d_backward_kernel(
      grad_output, output_size, input_size, align_corners, scales, grad_input);
  return grad_input;
}

} // namespace at
