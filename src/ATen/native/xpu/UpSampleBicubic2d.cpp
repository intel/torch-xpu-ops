#include <ATen/Context.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <aten/sycl/UpSampleBicubic2dKernels.h>

namespace at {
Tensor& XPUNativeFunctions::upsample_bicubic2d_out(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    Tensor& output) {
  native::xpu::upsample_bicubic2d_out_kernel(
      output, self, output_size, align_corners, scales_h, scales_w);
  return output;
}
} // namespace at