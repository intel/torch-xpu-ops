/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void adaptive_avg_pool2d_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input);

TORCH_XPU_API void adaptive_avg_pool2d_kernel(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size);

} // namespace at::native::xpu
