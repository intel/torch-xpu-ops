
#include <ATen/core/op_registration/adaption.h>
#include <ATen/div_rtn.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <ATen/native/xpu/sycl/Col2ImKernel.h>

#include <comm/xpu_aten.h>
#include <xpu/ATen/ops/col2im_native.h>

namespace at::native {

Tensor& col2im_out_xpu(
    const Tensor& self,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::col2im_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::col2im_out", "self");
  at::native::xpu::col2im_kernel(
      out, self, output_size, kernel_size, dilation, padding, stride);
  return out;
}

Tensor col2im_xpu(
    const Tensor& self,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::col2im", "self");
  Tensor output = at::empty_like(self);
  at::native::xpu::col2im_kernel(
      output, self, output_size, kernel_size, dilation, padding, stride);
  return output;
}

} // namespace at::native
