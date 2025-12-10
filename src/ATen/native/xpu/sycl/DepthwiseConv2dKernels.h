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
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void conv_depthwise2d_forward_kernel(
    const Tensor& input,
    const Tensor& output,
    const Tensor& weight,
    const Tensor& bias,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH);

TORCH_XPU_API void conv_depthwise2d_backward_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& grad_input,
    const Tensor& weight,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH);

TORCH_XPU_API void conv_depthwise2d_grad_weight_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& grad_weight,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH);

} // namespace at::native::xpu
