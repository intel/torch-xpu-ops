#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/UpSampleBilinear2dKernels.h>
#include <comm/RegisterUtils.h>

#include <xpu/ATen/ops/upsample_bilinear2d_backward_native.h>
#include <xpu/ATen/ops/upsample_bilinear2d_native.h>

namespace at {
namespace native {
TORCH_IMPL_FUNC(upsample_bilinear2d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  xpu::upsample_bilinear2d_out_kernel(
      output, input, output_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  xpu::upsample_bilinear2d_backward_out_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      align_corners,
      scales_h,
      scales_w);
}

TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_out_xpu) (
  const Tensor& input,
  IntArrayRef output_size,
  bool align_corners,
  std::optional<double> scales_h,
  std::optional<double> scales_w,
  const Tensor& output) {
    xpu::upsample_gen2d_aa_out_cuda_template<upsample_antialias::BilinearFilterFunctor>(
        output, input, output_size, align_corners, scales_h, scales_w);
}

} // namespace native
} // namespace at
