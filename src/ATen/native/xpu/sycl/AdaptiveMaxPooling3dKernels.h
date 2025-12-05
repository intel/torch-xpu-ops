/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void adaptive_max_pool3d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    const Tensor& output,
    const Tensor& indices);

TORCH_XPU_API void adaptive_max_pool3d_backward_kernel(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& gradInput);

} // namespace at::native::xpu
