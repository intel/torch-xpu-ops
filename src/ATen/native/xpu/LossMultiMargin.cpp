#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/MultiMarginLossKernels.h>
#include <ATen/xpu/XPUNativeFunctions.h>
namespace at {

Tensor& XPUNativeFunctions::multi_margin_loss_out(
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& out) {
  at::native::xpu::multi_margin_loss_kernel(
      self, target, p, margin, weight, reduction, out);
  return out;
}

Tensor XPUNativeFunctions::multi_margin_loss(
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<Tensor>& weight,
    int64_t reduction) {
  auto out = at::empty({0}, self.options());
  at::native::xpu::multi_margin_loss_kernel(
      self, target, p, margin, weight, reduction, out);
  return out;
}

Tensor& XPUNativeFunctions::multi_margin_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& grad_input) {
  at::native::xpu::multi_margin_loss_backward_kernel(
      grad_output, self, target, p, margin, weight, reduction, grad_input);
  return grad_input;
}
Tensor XPUNativeFunctions::multi_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const c10::optional<Tensor>& weight,
    int64_t reduction) {
  auto grad_input = at::empty({0}, self.options());
  at::native::xpu::multi_margin_loss_backward_kernel(
      grad_output, self, target, p, margin, weight, reduction, grad_input);
  return grad_input;
}

} // namespace at