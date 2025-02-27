
#include <ATen/core/op_registration/adaption.h>
#include <ATen/div_rtn.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <ATen/ops/im2col_native.h>

#include <ATen/native/xpu/sycl/Im2ColKernel.h>
#include <comm/xpu_aten.h>

namespace at::native {
Tensor& im2col_out_xpu(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::im2col_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::im2col_out", "self");
  at::native::xpu::im2col_kernel(
      out, self, kernel_size, dilation, padding, stride);
  return out;
}

Tensor im2col_xpu(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::im2col", "self");
  Tensor output = at::empty_like(self);
  at::native::xpu::im2col_kernel(
      output, self, kernel_size, dilation, padding, stride);
  return output;
}
} // namespace at::native
