/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/FusedAdamKernels.h>
#include <ATen/native/xpu/sycl/FusedAdamUtils.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

namespace at::native::xpu {

void fused_adam_kernel(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  fused_adam_kernel_common<ADAM_MODE::ORIGINAL, /*amsgrad=*/false>(
      params,
      grads,
      exp_avgs,
      exp_avg_sqs,
      /*max_exp_avg_sqs=*/{},
      state_steps,
      /*lr_ptr=*/nullptr,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

void fused_adam_kernel(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  fused_adam_kernel_common<ADAM_MODE::ORIGINAL, /*amsgrad=*/false>(
      params,
      grads,
      exp_avgs,
      exp_avg_sqs,
      /*max_exp_avg_sqs=*/{},
      state_steps,
      /*lr_ptr=*/lr.const_data_ptr<float>(),
      /*lr=*/1.0,
      beta1,
      beta2,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

} // namespace at::native::xpu
