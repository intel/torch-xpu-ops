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

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void _softmax_kernel(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    const Tensor& output);

TORCH_XPU_API void _log_softmax_kernel(
    const Tensor& input,
    const int64_t dim,
    const bool half_to_float,
    const Tensor& output);

TORCH_XPU_API void _softmax_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool half_to_float,
    const Tensor& grad_input);

TORCH_XPU_API void _log_softmax_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool half_to_float,
    const Tensor& grad_input);

TORCH_XPU_API Tensor
_safe_softmax_kernel(const Tensor& self, int64_t dim, const bool half_to_float);

TORCH_XPU_API Tensor masked_softmax_kernel(
    const Tensor& input_,
    const Tensor& mask_,
    const std::optional<int64_t> dim_,
    const std::optional<int64_t> mask_type_);

TORCH_XPU_API Tensor masked_softmax_backward_kernel(
    const Tensor& grad_,
    const Tensor& output_,
    const Tensor& mask_,
    const std::optional<int64_t> dim_);

} // namespace xpu
} // namespace native
} // namespace at
