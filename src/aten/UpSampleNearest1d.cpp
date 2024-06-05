#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <aten/sycl/UpSampleNearest1d.h>

namespace at {
// using namespace at::native;
Tensor& upsample_nearest_meta(
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

  output = at::empty({nbatch, channels, output_width}, input.options());
  return output;
}
Tensor XPUNativeFunctions::_upsample_nearest_exact1d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  printf("upsample_ meng1\n");
  Tensor output;
  output = upsample_nearest_meta(input, output, output_size);
  at::native::xpu::upsample_nearest1d_out_kernel(
      output, input, output_size, scales, true);
  return output;
}

Tensor& XPUNativeFunctions::_upsample_nearest_exact1d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    Tensor& output) {
  printf("upsample_ meng2\n");
  at::native::xpu::upsample_nearest1d_out_kernel(
      output, input, output_size, scales, true);
  return output;
}

Tensor XPUNativeFunctions::upsample_nearest1d(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales) {
  printf("upsample_ meng3\n");
  Tensor output;
  output = upsample_nearest_meta(input, output, output_size);
  at::native::xpu::upsample_nearest1d_out_kernel(
      output, input, output_size, scales, false);
  return output;
}

Tensor& XPUNativeFunctions::upsample_nearest1d_out(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales,
    Tensor& output) {
  printf("upsample_ meng4\n");
  at::native::xpu::upsample_nearest1d_out_kernel(
      output, input, output_size, scales, false);
  return output;
}
} // namespace at