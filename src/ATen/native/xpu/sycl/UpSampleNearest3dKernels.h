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

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API void upsample_nearest3d_kernel(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    bool is_exact);

TORCH_XPU_API void upsample_nearest3d_backward_kernel(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales_d,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    bool is_exact);

} // namespace at::native::xpu
