#include <ATen/ATen.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl//UpSampleNearest1dKernels.h>

namespace at {
// using namespace at::native;
Tensor& upsample_nearest1d_meta(
    const Tensor& input,
    Tensor& output,
    IntArrayRef output_size) {
  auto input_size = input.sizes();
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  int64_t output_width = output_size[0];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");
  TORCH_CHECK(
      (input.size(1) != 0 && input.size(2) != 0) && input.dim() == 3,
      "Non-empty 3D data tensor expected but got a tensor with sizes ",
      input.sizes());

  if (!output.defined())
    output = at::empty({nbatch, channels, output_width}, input.options());
  return output;
}

Tensor& upsample_nearest1d_backward_meta(
    const Tensor& grad_output,
    Tensor& grad_input,
    IntArrayRef input_size,
    IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());
  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());
  int64_t output_width = output_size[0];
  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];
  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");
  check_dim_size(grad_output, 3, 0, nbatch);
  check_dim_size(grad_output, 3, 1, channels);
  check_dim_size(grad_output, 3, 2, output_width);
  if (!grad_input.defined())
    grad_input = at::empty(input_size, grad_output.options());
  return grad_input;
}

Tensor XPUNativeFunctions::_upsample_nearest_exact1d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  Tensor output;
  output = upsample_nearest1d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest1d_kernel(
      output, input, output_size, scales, true);
  return output;
}

Tensor& XPUNativeFunctions::_upsample_nearest_exact1d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    Tensor& output) {
  upsample_nearest1d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest1d_kernel(
      output, input, output_size, scales, true);
  return output;
}

Tensor XPUNativeFunctions::upsample_nearest1d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  Tensor output;
  output = upsample_nearest1d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest1d_kernel(
      output, input, output_size, scales, false);
  return output;
}

Tensor& XPUNativeFunctions::upsample_nearest1d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    Tensor& output) {
  upsample_nearest1d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest1d_kernel(
      output, input, output_size, scales, false);
  return output;
}

Tensor XPUNativeFunctions::_upsample_nearest_exact1d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales) {
  Tensor grad_input;
  grad_input = upsample_nearest1d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest1d_backward_kernel(
      grad_input, grad_output, output_size, input_size, scales, true);
  return grad_input;
}
Tensor& XPUNativeFunctions::_upsample_nearest_exact1d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    Tensor& grad_input) {
  upsample_nearest1d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest1d_backward_kernel(
      grad_input, grad_output, output_size, input_size, scales, true);
  return grad_input;
}

Tensor XPUNativeFunctions::upsample_nearest1d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales) {
  Tensor grad_input;
  grad_input = upsample_nearest1d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest1d_backward_kernel(
      grad_input, grad_output, output_size, input_size, scales, false);
  return grad_input;
}

Tensor& XPUNativeFunctions::upsample_nearest1d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    Tensor& grad_input) {
  upsample_nearest1d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest1d_backward_kernel(
      grad_input, grad_output, output_size, input_size, scales, true);
  return grad_input;
}

} // namespace at
