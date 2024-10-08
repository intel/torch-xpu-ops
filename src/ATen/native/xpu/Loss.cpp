#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/xpu/sycl/BinaryMiscOpsKernels.h>
#include <ATen/native/xpu/sycl/LossKernels.h>
#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/zeros_like.h>

#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(mse_stub, &xpu::mse_kernel);
REGISTER_XPU_DISPATCH(mse_backward_stub, &xpu::mse_backward_kernel);
REGISTER_XPU_DISPATCH(huber_stub, &xpu::huber_kernel);
REGISTER_XPU_DISPATCH(huber_backward_stub, &xpu::huber_backward_kernel);
REGISTER_XPU_DISPATCH(smooth_l1_stub, &xpu::smooth_l1_kernel);
REGISTER_XPU_DISPATCH(smooth_l1_backward_stub, &xpu::smooth_l1_backward_kernel);

Tensor binary_cross_entropy_xpu(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  Tensor loss = at::empty_like(self);
  return native::xpu::binary_cross_entropy_kernel(
      self, target, weight, reduction, loss);
}

Tensor& binary_cross_entropy_out_xpu(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    Tensor& loss) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  return native::xpu::binary_cross_entropy_kernel(
      self, target, weight, reduction, loss);
}

Tensor binary_cross_entropy_backward_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  Tensor grad_input = at::empty_like(self);
  return native::xpu::binary_cross_entropy_backward_kernel(
      grad_output, self, target, weight, reduction, grad_input);
}

Tensor& binary_cross_entropy_backward_out_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    Tensor& grad_input) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  return native::xpu::binary_cross_entropy_backward_kernel(
      grad_output, self, target, weight, reduction, grad_input);
}

static inline at::Tensor apply_loss_reduction(
    const at::Tensor& unreduced,
    int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

Tensor soft_margin_loss_xpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor loss;
  auto iter = TensorIterator::binary_op(loss, self, target);
  xpu::soft_margin_kernel(iter);
  return apply_loss_reduction(iter.output(), reduction);
}

Tensor& soft_margin_loss_out_xpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& out) {
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::binary_op(loss, self, target);
    xpu::soft_margin_kernel(iter);
    if (reduction == Reduction::Mean) {
      at::mean_out(
          out,
          iter.output(),
          OptionalIntArrayRef{IntArrayRef{}},
          false,
          c10::nullopt);
    } else {
      at::sum_out(
          out,
          iter.output(),
          OptionalIntArrayRef{IntArrayRef{}},
          false,
          c10::nullopt);
    }
  } else {
    auto iter = TensorIterator::binary_op(out, self, target);
    xpu::soft_margin_kernel(iter);
  }
  return out;
}

Tensor& soft_margin_loss_backward_out_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / self.numel() : 1.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(target)
                  .add_input(grad_output)
                  .build();
  xpu::soft_margin_backward_kernel(iter, norm);
  return grad_input;
}

Tensor soft_margin_loss_backward_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor grad_input = at::zeros_like(
      self, self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  return soft_margin_loss_backward_out_xpu(
      grad_output, self, target, reduction, grad_input);
}

} // namespace native
} // namespace at
