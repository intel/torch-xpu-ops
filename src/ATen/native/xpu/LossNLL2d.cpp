#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/LossNLL2dKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {
std::tuple<Tensor, Tensor> XPUNativeFunctions::nll_loss2d_forward(
    const Tensor& self,
    const Tensor& target,
    const ::std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto output = at::empty({0}, self.options()); // zhy 0?or {}
  auto total_weight = at::empty({0}, self.options());
  native::xpu::nll_loss2d_forward_out_kernel(
      output, total_weight, self, target, weight, reduction, ignore_index);

  return std::make_tuple(output, total_weight);
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::nll_loss2d_forward_out(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  native::xpu::nll_loss2d_forward_out_kernel(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

Tensor XPUNativeFunctions::nll_loss2d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const ::std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  auto grad_input = at::empty_like(self);
  native::xpu::nll_loss2d_backward_out_kernel(
      grad_input,
      grad_output,
      self,
      target,
      weight_opt,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

Tensor& XPUNativeFunctions::nll_loss2d_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const ::std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    Tensor& grad_input) {
  native::xpu::nll_loss2d_backward_out_kernel(
      grad_input,
      grad_output,
      self,
      target,
      weight_opt,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

} // namespace at