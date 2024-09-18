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
  printf("multilabel_margin_loss_forward\n");
  at::native::xpu::multilabel_margin_loss_kernel(
      self, target, reduction, output, is_target);
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::multilabel_margin_loss_forward(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  printf("multilabel_margin_loss_forward3\n");
  auto output = at::empty({0}, self.options());
  auto is_target = at::empty({0}, self.options());
  printf("multilabel_margin_loss_forward4\n");
  at::native::xpu::multilabel_margin_loss_kernel(
      self, target, reduction, output, is_target);
  printf("multilabel_margin_loss_forward5\n");
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

Tensor& XPUNativeFunctions::multilabel_margin_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  printf("multilabel_margin_loss_backward_out\n");
  at::native::xpu::multilabel_margin_loss_backward_kernel(
      grad_output, self, target, reduction, is_target, grad_input);
  printf("multilabel_margin_loss_backward_out done\n");
  return grad_input;
}

Tensor XPUNativeFunctions::multilabel_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  printf("multilabel_margin_loss_backward_out2\n");
  auto grad_input = zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::xpu::multilabel_margin_loss_backward_kernel(
      grad_output, self, target, reduction, is_target, grad_input);
  printf("multilabel_margin_loss_backward_out2 done\n");
  return grad_input;
}

} // namespace at
