#include <ATen/Context.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <aten/sycl/UpSampleNearest2dKernels.h>

namespace at {
Tensor& XPUNativeFunctions::upsample_nearest2d_out(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  native::xpu::upsample_nearest2d_out_kernel(
      output, self, output_size, scales_h, scales_w);
  return output;
}

Tensor& XPUNativeFunctions::upsample_nearest2d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  native::xpu::upsample_nearest2d_backward_out_kernel(
      grad_input, grad_output, output_size, input_size, scales_h, scales_w);
  return grad_input;
}
} // namespace at