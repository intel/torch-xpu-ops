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

TORCH_XPU_API Tensor& multi_margin_loss_kernel(
    const Tensor& input,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& out);

TORCH_XPU_API Tensor& multi_margin_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Scalar& p,
    const Scalar& margin,
    const std::optional<Tensor>& weight,
    int64_t reduction,
    Tensor& grad_input);

} // namespace at::native::xpu
