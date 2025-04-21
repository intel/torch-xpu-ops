#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/xpu/sycl/FractionalMaxPool3dKernels.h>
#include <ATen/ops/empty.h>

#include <xpu/ATen/ops/fractional_max_pool3d_backward_native.h>
#include <xpu/ATen/ops/fractional_max_pool3d_native.h>

namespace at::native {

TORCH_IMPL_FUNC(fractional_max_pool3d_out_xpu)
(const Tensor& input,
 int64_t poolSizeT,
 int64_t poolSizeH,
 int64_t poolSizeW,
 int64_t outputT,
 int64_t outputH,
 int64_t outputW,
 const Tensor& randomSamples,
 int64_t numBatch,
 int64_t numPlanes,
 int64_t inputT,
 int64_t inputH,
 int64_t inputW,
 const Tensor& output,
 const Tensor& indices) {
  xpu::fractional_max_pool3d_kernel(
      input,
      poolSizeT,
      poolSizeH,
      poolSizeW,
      outputT,
      outputH,
      outputW,
      randomSamples,
      numBatch,
      numPlanes,
      inputT,
      inputH,
      inputW,
      output,
      indices);
}

Tensor& fractional_max_pool3d_backward_out_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const Tensor& indices,
    Tensor& grad_input) {
  globalContext().alertNotDeterministic(
      "fractional_max_pool3d_backward_out_xpu");
  xpu::fractional_max_pool3d_backward_kernel(
      grad_input, grad_output, input, output_size, indices);
  return grad_input;
}

Tensor fractional_max_pool3d_backward_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const Tensor& indices) {
  globalContext().alertNotDeterministic("fractional_max_pool3d_backward_xpu");
  Tensor grad_input = at::empty({0}, input.options());
  xpu::fractional_max_pool3d_backward_kernel(
      grad_input, grad_output, input, output_size, indices);
  return grad_input;
}

} // namespace at::native
