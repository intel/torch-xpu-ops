#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/div_rtn.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <aten/sycl/Im2ColKernel.h>

namespace at {

Tensor& XPUNativeFunctions::im2col_out(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& out) {
  at::native::xpu::im2col_out_template(
      out, self, kernel_size, dilation, padding, stride);
  return out;
}

Tensor XPUNativeFunctions::im2col(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor output = at::empty_like(self);
  at::native::xpu::im2col_out_template(
      output, self, kernel_size, dilation, padding, stride);
  return output;
}

} // namespace at
