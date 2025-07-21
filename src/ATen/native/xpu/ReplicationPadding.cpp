#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Padding.h>
#include <ATen/native/xpu/sycl/ReplicationPaddingKernels.h>

#include <comm/RegisterUtils.h>

#include <ATen/ops/replication_pad1d_backward_native.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad2d_backward_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad3d_backward_native.h>
#include <ATen/ops/replication_pad3d_native.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(replication_pad1d_out_xpu)
(const Tensor& input, IntArrayRef paddingSize, const Tensor& output) {
  xpu::replication_pad1d_kernel(output, input, paddingSize);
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_xpu)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef paddingSize,
 const Tensor& gradInput) {
  xpu::replication_pad1d_backward_kernel(
      gradInput, gradOutput, input, paddingSize);
}

TORCH_IMPL_FUNC(replication_pad2d_out_xpu)
(const Tensor& input, IntArrayRef paddingSize, const Tensor& output) {
  xpu::replication_pad2d_kernel(output, input, paddingSize);
}

Tensor& replication_pad2d_backward_out_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  xpu::replication_pad2d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor replication_pad2d_backward_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  xpu::replication_pad2d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

TORCH_IMPL_FUNC(replication_pad3d_out_xpu)
(const Tensor& input, IntArrayRef paddingSize, const Tensor& output) {
  xpu::replication_pad3d_kernel(output, input, paddingSize);
}

Tensor replication_pad3d_backward_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    at::IntArrayRef padding) {
  auto grad_input = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  xpu::replication_pad3d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

Tensor& replication_pad3d_backward_out_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding,
    Tensor& grad_input) {
  xpu::replication_pad3d_backward_kernel(
      grad_input, grad_output, input, padding);
  return grad_input;
}

} // namespace native
} // namespace at