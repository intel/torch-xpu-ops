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

TORCH_XPU_API void adaptive_avg_pool2d_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input);

TORCH_XPU_API void adaptive_avg_pool2d_kernel(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size);

} // namespace at::native::xpu
