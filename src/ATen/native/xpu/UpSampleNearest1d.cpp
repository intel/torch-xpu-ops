#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/UpSampleNearest1dKernels.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/upsample_nearest1d_native.h>

namespace at {
namespace native {
TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales,
 const Tensor& output) {
  xpu::upsample_nearest1d_kernel(output, input, output_size, scales, true);
}

TORCH_IMPL_FUNC(upsample_nearest1d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales,
 const Tensor& output) {
  xpu::upsample_nearest1d_kernel(output, input, output_size, scales, false);
}

TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales,
 const Tensor& grad_input) {
  grad_input.zero_();
  xpu::upsample_nearest1d_backward_kernel(
      grad_input, grad_output, output_size, input_size, scales, true);
}

TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales,
 const Tensor& grad_input) {
  grad_input.zero_();
  xpu::upsample_nearest1d_backward_kernel(
      grad_input, grad_output, output_size, input_size, scales, false);
}

} // namespace native
} // namespace at
