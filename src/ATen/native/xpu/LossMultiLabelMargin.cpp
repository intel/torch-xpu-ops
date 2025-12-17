/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/sycl/MultiLabelMarginLossKernels.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/multilabel_margin_loss.h>
#include <ATen/ops/zeros_like.h>

namespace at {
namespace native {

std::tuple<Tensor&, Tensor&> multilabel_margin_loss_forward_out_xpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  xpu::multilabel_margin_loss_kernel(
      self, target, reduction, output, is_target);
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_xpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto output = at::empty({0}, self.options());
  auto is_target = at::empty({0}, self.options());
  xpu::multilabel_margin_loss_kernel(
      self, target, reduction, output, is_target);
  return std::make_tuple(output, is_target);
}

Tensor& multilabel_margin_loss_backward_xpu_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  xpu::multilabel_margin_loss_backward_kernel(
      grad_output, self, target, reduction, is_target, grad_input);
  return grad_input;
}

Tensor multilabel_margin_loss_backward_xpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  auto grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  xpu::multilabel_margin_loss_backward_kernel(
      grad_output, self, target, reduction, is_target, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at
