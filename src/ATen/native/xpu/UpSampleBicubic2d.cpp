#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/UpSampleBicubic2dKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>

#include <ATen/xpu/ops/upsample_bicubic2d_native.h>
namespace at {
namespace native {
TORCH_IMPL_FUNC(upsample_bicubic2d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  xpu::upsample_bicubic2d_kernel(
      output, input, output_size, align_corners, scales_h, scales_w);
}
} // namespace native
} // namespace at
