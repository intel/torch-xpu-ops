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
namespace at::native::xpu {

TORCH_XPU_API Tensor deform_conv2d_forward_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& mask,
    const Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
deform_conv2d_backward_kernel(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& offset,
    const Tensor& mask,
    const Tensor& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t n_weight_grps,
    int64_t n_offset_grps,
    bool use_mask);
} // namespace at::native::xpu