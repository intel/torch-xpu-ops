/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor& binary_cross_entropy_kernel(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& loss);

TORCH_XPU_API Tensor& binary_cross_entropy_backward_kernel(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    Tensor& grad_input);

} // namespace at::native::xpu
