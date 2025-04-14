#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/LossNLLKernel.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

#include <xpu/ATen/ops/nll_loss_backward_native.h>
#include <xpu/ATen/ops/nll_loss_forward_native.h>

namespace at {
namespace native {
TORCH_IMPL_FUNC(nll_loss_forward_out_xpu)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  xpu::nll_loss_forward_kernel(
      self,
      target,
      ((weight_opt.has_value() && (*weight_opt).defined())
           ? at::OptionalTensorRef(*weight_opt)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      output,
      total_weight);
}

TORCH_IMPL_FUNC(nll_loss_backward_out_xpu)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input) {
  grad_input.zero_();
  xpu::nll_loss_backward_kernel(
      grad_output,
      self,
      target,
      ((weight_opt.has_value() && (*weight_opt).defined())
           ? at::OptionalTensorRef(*weight_opt)
           : at::OptionalTensorRef()),
      reduction,
      ignore_index,
      total_weight,
      grad_input);
}

} // namespace native
} // namespace at