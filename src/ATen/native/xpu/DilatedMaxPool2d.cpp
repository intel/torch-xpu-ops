#include <ATen/core/Tensor.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xpu/sycl/DilatedMaxPool2d.h>
#include <comm/RegisterUtils.h>

#include <xpu/ATen/ops/max.h>
#include <xpu/ATen/ops/max_pool2d_with_indices_backward_native.h>
#include <xpu/ATen/ops/max_pool2d_with_indices_native.h>

namespace at {
namespace native {
TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_xpu)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& indices,
 const Tensor& gradInput) {
  xpu::max_pool2d_with_indices_backward_kernel(
      gradInput,
      gradOutput,
      input,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_out_xpu)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& output,
 const Tensor& indices) {
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = output.size(-2);
  const int64_t outputWidth = output.size(-1);
  if (outputHeight == 1 && outputWidth == 1 && inputHeight <= kH &&
      inputWidth <= kW && padH == 0 && padW == 0) {
    auto smf = input.suggest_memory_format();
    Tensor input_ = input.contiguous(smf);
    bool is_3d = input.ndimension() == 3;
    Tensor indices_, output_;
    if (is_3d) {
      indices_ = indices.contiguous();
      output_ = output.contiguous();
    } else {
      indices_ = indices.contiguous(smf);
      output_ = output.contiguous(smf);
    }
    if (!is_3d) {
      input_.resize_({nbatch, nInputPlane, 1, inputHeight * inputWidth}, smf);
      output_.resize_(
          {nbatch, nInputPlane, 1, outputHeight * outputWidth}, smf);
      indices_.resize_(
          {nbatch, nInputPlane, 1, outputHeight * outputWidth}, smf);
      at::max_outf(input_, 3, true, output_, indices_);
    } else {
      at::max_outf(input_, 2, true, output_, indices_);
    }

    if (!is_3d) {
      input_.resize_({nbatch, nInputPlane, inputHeight, inputWidth}, smf);
      output_.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
      indices_.resize_({nbatch, nInputPlane, outputHeight, outputWidth}, smf);
    }

    if ((is_3d && !indices.is_contiguous()) ||
        (!is_3d && !indices.is_contiguous(smf))) {
      indices.copy_(indices_);
    }

    if ((is_3d && !output.is_contiguous()) ||
        (!is_3d && !output.is_contiguous(smf))) {
      output.copy_(output_);
    }
    return;
  }
  xpu::max_pool2d_with_indices_kernel(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);
}
} // namespace native
} // namespace at
