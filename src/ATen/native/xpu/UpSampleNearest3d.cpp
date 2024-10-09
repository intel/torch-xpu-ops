#include <ATen/ATen.h>
#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/UpSampleNearest3dKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

namespace at {

Tensor& upsample_nearest3d_meta(
    const Tensor& input,
    Tensor& output,
    IntArrayRef output_size) {
  auto full_output_size =
      native::xpu::upsample_3d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input.sizes());
  if (output.defined()) {
    at::xpu::resize_out(
        output,
        full_output_size,
        {},
        input.options().memory_format(input.suggest_memory_format()));
  } else {
    output = at::xpu::create_out(
        full_output_size,
        {},
        input.options().memory_format(input.suggest_memory_format()));
  }
  return output;
}

Tensor& upsample_nearest3d_backward_meta(
    const Tensor& grad_output,
    Tensor& grad_input,
    IntArrayRef input_size,
    IntArrayRef output_size) {
  auto full_output_size =
      native::xpu::upsample_3d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 5,
      "Expected grad_output to be a tensor of dimension 5 but got: dimension ",
      grad_output.dim());

  for (const auto i : c10::irange(5)) {
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

  if (grad_input.defined()) {
    at::xpu::resize_out(grad_input, input_size, {}, grad_output.options());
  } else {
    grad_input = at::xpu::create_out(input_size, {}, grad_output.options());
  }
  return grad_input;
}

Tensor XPUNativeFunctions::_upsample_nearest_exact3d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor output;
  output = upsample_nearest3d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest3d_kernel(
      output, input, output_size, scales_d, scales_h, scales_w, true);
  return output;
}

Tensor& XPUNativeFunctions::_upsample_nearest_exact3d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  upsample_nearest3d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest3d_kernel(
      output, input, output_size, scales_d, scales_h, scales_w, true);
  return output;
}

Tensor XPUNativeFunctions::upsample_nearest3d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor output;
  output = upsample_nearest3d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest3d_kernel(
      output, input, output_size, scales_d, scales_h, scales_w, false);
  return output;
}

Tensor& XPUNativeFunctions::upsample_nearest3d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  upsample_nearest3d_meta(input, output, output_size);
  at::native::xpu::upsample_nearest3d_kernel(
      output, input, output_size, scales_d, scales_h, scales_w, false);
  return output;
}

Tensor XPUNativeFunctions::_upsample_nearest_exact3d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor grad_input;
  grad_input = upsample_nearest3d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest3d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_d,
      scales_h,
      scales_w,
      true);
  return grad_input;
}

Tensor& XPUNativeFunctions::_upsample_nearest_exact3d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  upsample_nearest3d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest3d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_d,
      scales_h,
      scales_w,
      true);
  return grad_input;
}

Tensor XPUNativeFunctions::upsample_nearest3d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor grad_input;
  grad_input = upsample_nearest3d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest3d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_d,
      scales_h,
      scales_w,
      false);
  return grad_input;
}

Tensor& XPUNativeFunctions::upsample_nearest3d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  upsample_nearest3d_backward_meta(
      grad_output, grad_input, input_size, output_size);
  at::native::xpu::upsample_nearest3d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_d,
      scales_h,
      scales_w,
      false);
  return grad_input;
}

} // namespace at
