/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/ForeachUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adagrad.h>
#include <ATen/ops/_fused_adagrad_native.h>
#endif

#include <ATen/native/xpu/sycl/FusedAdagradKernels.h>

namespace at {
namespace native {

void _fused_adagrad_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(
      at::native::check_fast_path_restrictions({params, grads, state_sums}),
      "params, grads, and state_sums must have same dtype, device, and layout");
  xpu::fused_adagrad_kernel(
      params,
      grads,
      state_sums,
      state_steps,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

void _fused_adagrad_kernel_xpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adagrad_kernel_xpu_(
        params,
        grads,
        state_sums,
        state_steps,
        lr.item<double>(),
        lr_decay,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions({params, grads, state_sums}),
        "params, grads, and state_sums must have same dtype, device, and layout");
    xpu::fused_adagrad_kernel(
        params,
        grads,
        state_sums,
        state_steps,
        lr,
        lr_decay,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

} // namespace native
} // namespace at
