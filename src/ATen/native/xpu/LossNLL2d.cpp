/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/LossNLL2dKernels.h>

namespace at {
namespace native {

namespace {
void check_inputs_nll_loss2d(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight) {
  TORCH_CHECK(
      target.dim() == 3,
      "only batches of spatial targets supported (3D tensors)"
      " but got targets of size: : ",
      target.sizes());
  TORCH_CHECK(
      input.dim() == 4,
      "only batches of spatial inputs supported (4D tensors), "
      "but got input of size: ",
      input.sizes());
  TORCH_CHECK(
      !weight.defined() || weight.numel() == input.size(1),
      "weight tensor should be defined either for all or no classes");

  TORCH_CHECK(
      input.size(0) == target.size(0) && input.size(2) == target.size(1) &&
          input.size(3) == target.size(2),
      "input and target batch or spatial sizes don't match: target ",
      target.sizes(),
      ", input ",
      input.sizes());
}
} // namespace

std::tuple<Tensor, Tensor> nll_loss2d_forward_xpu(
    const Tensor& self,
    const Tensor& target,
    const ::std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  check_inputs_nll_loss2d(self, target, weight);

  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  native::xpu::nll_loss2d_forward_kernel(
      output, total_weight, self, target, weight, reduction, ignore_index);

  return std::make_tuple(output, total_weight);
}

std::tuple<Tensor&, Tensor&> nll_loss2d_forward_out_xpu(
    const Tensor& self,
    const Tensor& target,
    const std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  check_inputs_nll_loss2d(self, target, weight);

  native::xpu::nll_loss2d_forward_kernel(
      output, total_weight, self, target, weight, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

Tensor nll_loss2d_backward_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const ::std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  check_inputs_nll_loss2d(self, target, weight);

  auto grad_input = at::empty_like(self);
  native::xpu::nll_loss2d_backward_kernel(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

Tensor& nll_loss2d_backward_out_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const ::std::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    Tensor& grad_input) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  check_inputs_nll_loss2d(self, target, weight);
  native::xpu::nll_loss2d_backward_kernel(
      grad_input,
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight);
  return grad_input;
}

} // namespace native
} // namespace at