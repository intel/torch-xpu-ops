#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>

#include <aten/sycl/DilatedMaxPool2d.h>

namespace at {

Tensor& XPUNativeFunctions::max_pool2d_with_indices_backward_out(
    const Tensor& grad_output_,
    const Tensor& self_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices_,
    Tensor& grad_input) {
  /* PyTorch support two cases of MaxPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the sementics
     of Input (C, H, W) case. Then the suggest_memory_format can only be
     Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output, indices;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    indices = indices_.contiguous();
    grad_input.zero_();
  } else {
    auto smf = self_.suggest_memory_format();
    self = self_.contiguous(smf);
    grad_output = grad_output_.contiguous(smf);
    indices = indices_.contiguous(smf);
    grad_input.zero_();
  }
  at::native::xpu::max_pool2d_with_indices_backward_out_template(
      grad_input,
      grad_output,
      self,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
  return grad_input;
}

} // namespace at
