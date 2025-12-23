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

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void nll_loss2d_forward_kernel(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index);

TORCH_XPU_API void nll_loss2d_backward_kernel(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight);

} // namespace at::native::xpu
