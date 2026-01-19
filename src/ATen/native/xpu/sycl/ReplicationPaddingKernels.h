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

TORCH_XPU_API void replication_pad1d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad1d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad2d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad2d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad3d_kernel(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef padding);

TORCH_XPU_API void replication_pad3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef padding);

} // namespace at::native::xpu
