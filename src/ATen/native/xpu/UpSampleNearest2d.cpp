#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/UpSampleNearest2dKernels.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/_upsample_nearest_exact2d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/upsample_nearest2d_backward_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
namespace at {

namespace native {

TORCH_IMPL_FUNC(upsample_nearest2d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  xpu::upsample_nearest2d_kernel(
      output, input, output_size, scales_h, scales_w, false);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  xpu::upsample_nearest2d_kernel(
      output, input, output_size, scales_h, scales_w, true);
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  grad_input.zero_();
  xpu::upsample_nearest2d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_h,
      scales_w,
      false);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  grad_input.zero_();
  xpu::upsample_nearest2d_backward_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      scales_h,
      scales_w,
      true);
}

} // namespace native
} // namespace at
