/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/ATen.h>
#include <optional>

namespace sycltla {

bool is_conv_available();

// --- Conv2d ---
at::Tensor conv2d_fprop(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

at::Tensor conv2d_dgrad(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::IntArrayRef input_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

at::Tensor conv2d_wgrad(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

// --- Conv3d ---
at::Tensor conv3d_fprop(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

at::Tensor conv3d_dgrad(
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    at::IntArrayRef input_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

at::Tensor conv3d_wgrad(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups);

} // namespace sycltla
