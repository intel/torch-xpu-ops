/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void fused_sgd_kernel(
    at::TensorList params,
    at::TensorList grads,
    const double weight_decay,
    const double momentum,
    const float* lr_ptr,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr,
    const float* found_inf_ptr);

TORCH_XPU_API void fused_sgd_with_momentum_kernel(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const float* lr_ptr,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr,
    const float* found_inf_ptr);

} // namespace at::native::xpu
