/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void adaptive_max_pool2d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    const Tensor& output,
    const Tensor& indices);

TORCH_XPU_API void adaptive_max_pool2d_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& grad_input);

} // namespace at::native::xpu
