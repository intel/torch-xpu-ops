#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/UpSampleTrilinear3dKernels.h>
#include <comm/SYCLContext.h>

#include <xpu/ATen/ops/upsample_trilinear3d_backward_native.h>
#include <xpu/ATen/ops/upsample_trilinear3d_native.h>

namespace at {
namespace native {
TORCH_IMPL_FUNC(upsample_trilinear3d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  xpu::upsample_trilinear3d_out_kernel(
      output, input, output_size, align_corners, scales_d, scales_h, scales_w);
}

TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  globalContext().alertNotDeterministic(
      "upsample_trilinear3d_backward_out_xpu");
  xpu::upsample_trilinear3d_backward_out_kernel(
      grad_input,
      grad_output,
      output_size,
      input_size,
      align_corners,
      scales_d,
      scales_h,
      scales_w);
}

} // namespace native
} // namespace at
