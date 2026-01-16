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
