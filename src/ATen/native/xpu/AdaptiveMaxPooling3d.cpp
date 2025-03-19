#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xpu/sycl/AdaptiveMaxPooling3dKernels.h>

#include <ATen/ops/adaptive_max_pool3d_backward_native.h>
#include <ATen/ops/adaptive_max_pool3d_native.h>
#include <ATen/ops/empty.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(adaptive_max_pool3d_out_xpu)
(const Tensor& input,
 IntArrayRef output_size,
 const Tensor& output,
 const Tensor& indices) {
  TensorArg output_arg{output, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{input, "input", 3};
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  if (input.numel() == 0 || output.numel() == 0) {
    return;
  }
  native::xpu::adaptive_max_pool3d_kernel(input, output_size, output, indices);
}

TORCH_IMPL_FUNC(adaptive_max_pool3d_backward_out_xpu)
(const Tensor& gradOutput,
 const Tensor& input,
 const Tensor& indices,
 const Tensor& gradInput) {
  TensorArg grad_input_arg{gradInput, "grad_input", 1};
  TensorArg grad_output_arg{gradOutput, "grad_output", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};

  checkAllSameGPU(
      __func__, {grad_input_arg, grad_output_arg, input_arg, indices_arg});

  if (gradOutput.numel() == 0) {
    return;
  }

  native::xpu::adaptive_max_pool3d_backward_kernel(
      gradOutput, input, indices, gradInput);
}
} // namespace native
} // namespace at
