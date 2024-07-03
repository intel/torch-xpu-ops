#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/UpSampleBilinear2dKernels.h>
#include <comm/RegisterUtils.h>

namespace at {

void upsample_bilinear2d_meta(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    Tensor& output) {
  auto full_output_size =
      native::xpu::upsample_2d_common_check(input.sizes(), output_size);

  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      input.numel() != 0 ||
          c10::multiply_integers(
              input.sizes().begin() + 1, input.sizes().end()),
      "Non-empty 4D data tensor expected but got a tensor with sizes ",
      input.sizes());

  auto memory_format = input.suggest_memory_format();
  if (output.defined()) {
    xpu::resize_out(
        output,
        full_output_size,
        {},
        input.options().memory_format(memory_format));
  } else {
    output = at::xpu::create_out(
        full_output_size, {}, input.options().memory_format(memory_format));
  }
}

void upsample_bilinear2d_backward_meta(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    Tensor& grad_input) {
  auto full_output_size =
      native::xpu::upsample_2d_common_check(input_size, output_size);

  TORCH_CHECK(
      grad_output.dim() == 4,
      "Expected grad_output to be a tensor of dimension 4 but got: dimension ",
      grad_output.dim());

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

  auto memory_format = grad_output.suggest_memory_format();
  if (grad_input.defined()) {
    xpu::resize_out(
        grad_input,
        input_size,
        {},
        grad_output.options().memory_format(memory_format));
  } else {
    grad_input = at::xpu::create_out(
        input_size, {}, grad_output.options().memory_format(memory_format));
  }
}

Tensor& XPUNativeFunctions::upsample_bilinear2d_out(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    Tensor& output) {
  upsample_bilinear2d_meta(
      self, output_size, align_corners, scales_h, scales_w, output);
  native::xpu::upsample_bilinear2d_out_kernel(
      output, self, output_size, align_corners, scales_h, scales_w);
  return output;
}

Tensor XPUNativeFunctions::upsample_bilinear2d(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w) {
  Tensor output;
  upsample_bilinear2d_out(
      self, output_size, align_corners, scales_h, scales_w, output);
  return output;
}

Tensor& XPUNativeFunctions::upsample_bilinear2d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  globalContext().alertNotDeterministic("upsample_bilinear2d_backward_xpu");

  upsample_bilinear2d_backward_meta(
      grad_output,
      output_size,
      input_size,
      align_corners,
      scales_h,
      scales_w,
      grad_input);

  native::xpu::upsample_bilinear2d_backward_out_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      align_corners,
      scales_h,
      scales_w);
  return grad_input;
}

Tensor XPUNativeFunctions::upsample_bilinear2d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  Tensor grad_input;
  upsample_bilinear2d_backward_out(
      grad_output,
      output_size,
      input_size,
      align_corners,
      scales_h,
      scales_w,
      grad_input);
  return grad_input;
}

} // namespace at
