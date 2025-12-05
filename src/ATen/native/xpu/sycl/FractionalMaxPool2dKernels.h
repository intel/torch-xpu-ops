/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void fractional_max_pool2d_kernel(
    const Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const Tensor& randomSamples,
    const Tensor& output,
    const Tensor& indices);

TORCH_XPU_API void fractional_max_pool2d_backward_kernel(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef pool_size /* unused */,
    IntArrayRef output_size,
    const Tensor& indices,
    const Tensor& gradInput);

} // namespace at::native::xpu
