#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <aten/sycl/AdaptiveAveragePooling2d.h>

namespace at {

Tensor XPUNativeFunctions::_adaptive_avg_pool2d_backward(
    const Tensor& grad_output_,
    const Tensor& self_) {
  /* PyTorch support two cases of AdaptiveAvgPool2d:
     1. 3D: Input (C, H, W),  Output (C, H0, W0), Kernel (kH, kW)
     This case does not support channel last format. For a 3-dim tensor,
     the PyTorch suggest_memory_format can only be Contiguous or
     ChannelsLast1D (nwc), the ChannelsLast1D (nwc) does not match the
     sementics of Input (C, H, W) case. Then the suggest_memory_format can
     only be Contiguous.
     2. 4D: Input (N, C, H, W),  Output (N, C, H0, W0), Kernel (kH, kW)
     This case supports Contiguous and ChannelsLast2D memory_format. */
  Tensor self, grad_output, grad_input;
  if (self_.ndimension() == 3) {
    self = self_.contiguous();
    grad_output = grad_output_.contiguous();
    grad_input = at::empty_like(self);
  } else {
    auto smf = self_.suggest_memory_format();
    self = contiguous_if_needed(self_, smf);
    grad_output = contiguous_if_needed(grad_output_, smf);
    grad_input = at::empty_like(self_, smf);
  }

  adaptive_avg_pool2d_backward_out_template(grad_input, grad_output, self);
  return grad_input;
}
} // namespace at