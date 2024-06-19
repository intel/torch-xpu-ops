
#include <ATen/core/Tensor.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/xpu/sycl/DilatedMaxPool2d.h>
#include <comm/RegisterUtils.h>

#include <ATen/xpu/ops/max_pool2d_with_indices_backward_native.h>

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
  xpu::max_pool2d_with_indices_backward_out_kernel(
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
} // namespace native
} // namespace at
