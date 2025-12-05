/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void im2col_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride);

} // namespace at::native::xpu
