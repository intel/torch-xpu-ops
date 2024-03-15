#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/div_rtn.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <aten/sycl/Col2ImKernel.h>

namespace at {

Tensor& XPUNativeFunctions::col2im_out(
    const Tensor& self,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& out) {
  at::native::xpu::col2im_out_template(
      out, self, output_size, kernel_size, dilation, padding, stride);
  return out;
}

Tensor XPUNativeFunctions::col2im(
    const Tensor& self,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor output = at::empty_like(self);

  at::native::xpu::col2im_out_template(
      output, self, output_size, kernel_size, dilation, padding, stride);
  return output;
}

} // namespace at
