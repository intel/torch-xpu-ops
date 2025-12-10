/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/MultiMarginLossKernels.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/multi_margin_loss_backward_native.h>
#include <ATen/ops/multi_margin_loss_native.h>

namespace at::native {

Tensor& multi_margin_loss_xpu_out(
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& out) {
  xpu::multi_margin_loss_kernel(
      self, target, p, margin, weight, reduction, out);
  return out;
}

Tensor multi_margin_loss_xpu(
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction) {
  auto out = at::empty({0}, self.options());
  xpu::multi_margin_loss_kernel(
      self, target, p, margin, weight, reduction, out);
  return out;
}

Tensor& multi_margin_loss_xpu_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& grad_input) {
  xpu::multi_margin_loss_backward_kernel(
      grad_output, self, target, p, margin, weight, reduction, grad_input);
  return grad_input;
}

Tensor multi_margin_loss_xpu_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction) {
  auto grad_input = at::empty({0}, self.options());
  xpu::multi_margin_loss_backward_kernel(
      grad_output, self, target, p, margin, weight, reduction, grad_input);
  return grad_input;
}

} // namespace at::native
