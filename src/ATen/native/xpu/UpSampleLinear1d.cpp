#include <ATen/ATen.h>
#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/UpSampleLinear1dKernels.h>

#include <comm/RegisterUtils.h>
#include "ATen/core/ATen_fwd.h"

#include <ATen/ops/upsample_linear1d_backward_native.h>
#include <ATen/ops/upsample_linear1d_native.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(upsample_linear1d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales,
 const Tensor& output) {
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  xpu::upsample_linear1d_kernel(
      input, output_size, align_corners, scales, output);
}

TORCH_IMPL_FUNC(upsample_linear1d_backward_out_xpu)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales,
 const Tensor& grad_input) {
  TensorArg grad_output_arg{grad_output, "grad_output", 1},
      grad_input_arg{grad_input, "grad_input", 2};
  checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});
  xpu::upsample_linear1d_backward_kernel(
      grad_output, output_size, input_size, align_corners, scales, grad_input);
}
} // namespace native

} // namespace at
