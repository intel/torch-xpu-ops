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

TORCH_XPU_API Tensor conv_depthwise3d_kernel(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

TORCH_XPU_API std::tuple<Tensor&, Tensor&, Tensor&>
_depthwise_3d_backward_kernel(
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const std::array<bool, 3> output_mask);

} // namespace at::native::xpu
