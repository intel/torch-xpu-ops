#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/DeformConv2dKernels.h>
#include <comm/XPUGuard.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>
namespace at::native::xpu {

at::Tensor deform_conv2d(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask) {
  return deform_conv2d_forward_kernel(
      input,
      weight,
      offset,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      n_weight_grps,
      n_offset_grps,
      use_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_deform_conv2d_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask) {
  return deform_conv2d_backward_kernel(
      grad_out,
      input,
      weight,
      offset,
      mask,
      bias,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      dilation_h,
      dilation_w,
      n_weight_grps,
      n_offset_grps,
      use_mask);
}

} // namespace at::native::xpu
