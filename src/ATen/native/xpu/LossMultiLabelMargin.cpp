#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/MultiLabelMarginLossKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
namespace at {

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::
    multilabel_margin_loss_forward_out(
        const Tensor& self,
        const Tensor& target,
        int64_t reduction,
        Tensor& output,
        Tensor& is_target) {
  at::native::xpu::multilabel_margin_loss_kernel(
      self, target, reduction, output, is_target);
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::multilabel_margin_loss_forward(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto output = at::empty({0}, self.options());
  auto is_target = at::empty({0}, self.options());
  at::native::xpu::multilabel_margin_loss_kernel(
      self, target, reduction, output, is_target);
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

Tensor& XPUNativeFunctions::multilabel_margin_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  at::native::xpu::multilabel_margin_loss_backward_kernel(
      grad_output, self, target, reduction, is_target, grad_input);
  return grad_input;
}

Tensor XPUNativeFunctions::multilabel_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  auto grad_input = zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::xpu::multilabel_margin_loss_backward_kernel(
      grad_output, self, target, reduction, is_target, grad_input);
  return grad_input;
}

} // namespace at
