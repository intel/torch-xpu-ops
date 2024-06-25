#include <ATen/ATen.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl//UpSampleNearest2dKernels.h>

namespace at {
// using namespace at::native;
Tensor& upsample_nearest2d_meta(
    const Tensor& input,
    Tensor& output,
    IntArrayRef output_size) {
  auto input_size = input.sizes();

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  if (!output.defined())
    output = at::empty(
        {nbatch, channels, output_height, output_width},
        input.options().memory_format(input.suggest_memory_format()));
  return output;
}

Tensor& upsample_nearest2d_backward_meta(
    const Tensor& grad_output,
    Tensor& grad_input,
    IntArrayRef input_size,
    IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ",
      grad_output.dim());
  std::array<int64_t, 4> full_output_size = {
      nbatch, channels, output_height, output_width};
  for (const auto i : c10::irange(4)) {
    TORCH_CHECK(
        grad_output.size(i) == full_output_size[i],
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_size[i],
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output.size(i));
  }
  if (!grad_input.defined())
    grad_input = at::empty(
        input_size,
        grad_output.options().memory_format(
            grad_output.suggest_memory_format()));
  return grad_input;
}

Tensor XPUNativeFunctions::_upsample_nearest_exact2d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor output;
  output = upsample_nearest2d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest2d_kernel(
      output, input, output_size, scales_h, scales_w, true);
  return output;
}

Tensor& XPUNativeFunctions::_upsample_nearest_exact2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  upsample_nearest2d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest2d_kernel(
      output, input, output_size, scales_h, scales_w, true);
  return output;
}

Tensor XPUNativeFunctions::upsample_nearest2d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor output;
  output = upsample_nearest2d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest2d_kernel(
      output, input, output_size, scales_h, scales_w, false);
  return output;
}

Tensor& XPUNativeFunctions::upsample_nearest2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  upsample_nearest2d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest2d_kernel(
      output, input, output_size, scales_h, scales_w, false);
  return output;
}

Tensor XPUNativeFunctions::_upsample_nearest_exact2d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor grad_input;
  grad_input = upsample_nearest2d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest2d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_h,
      scales_w,
      true);
  return grad_input;
}
Tensor& XPUNativeFunctions::_upsample_nearest_exact2d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  upsample_nearest2d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest2d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_h,
      scales_w,
      true);
  return grad_input;
}

Tensor XPUNativeFunctions::upsample_nearest2d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor grad_input;
  grad_input = upsample_nearest2d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest2d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_h,
      scales_w,
      false);
  return grad_input;
}

Tensor& XPUNativeFunctions::upsample_nearest2d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  upsample_nearest2d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest2d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_h,
      scales_w,
      true);
  return grad_input;
}

} // namespace at
