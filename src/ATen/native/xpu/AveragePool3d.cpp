#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/AveragePool3dKernels.h>

#include <xpu/ATen/ops/avg_pool3d_backward_native.h>
#include <xpu/ATen/ops/avg_pool3d_native.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(avg_pool3d_out_xpu)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& output) {
  at::native::xpu::avg_pool3d_kernel(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      output);
}

TORCH_IMPL_FUNC(avg_pool3d_backward_out_xpu)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 std::optional<int64_t> divisor_override,
 const Tensor& gradInput) {
  at::native::xpu::avg_pool3d_backward_kernel(
      gradOutput,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override,
      gradInput);
}

} // namespace native
} // namespace at
