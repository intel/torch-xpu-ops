#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/xpu/sycl/FractionalMaxPool2dKernels.h>

#include <xpu/ATen/ops/fractional_max_pool2d_backward_native.h>
#include <xpu/ATen/ops/fractional_max_pool2d_native.h>

namespace at::native {

TORCH_IMPL_FUNC(fractional_max_pool2d_out_xpu)
(const Tensor& input,
 IntArrayRef pool_size,
 IntArrayRef output_size,
 const Tensor& randomSamples,
 const Tensor& output,
 const Tensor& indices) {
  at::native::xpu::fractional_max_pool2d_out_kernel(
      output, indices, input, pool_size, output_size, randomSamples);
}

TORCH_IMPL_FUNC(fractional_max_pool2d_backward_xpu)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef pool_size /* unused */,
 IntArrayRef output_size,
 const Tensor& indices,
 const Tensor& gradInput) {
  at::native::xpu::fractional_max_pool2d_backward_kernel(
      gradInput, gradOutput, input, pool_size, output_size, indices);
}

} // namespace at::native