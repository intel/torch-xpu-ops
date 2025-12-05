/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <ATen/ATen.h>
namespace at::native::xpu {

TORCH_XPU_API Tensor repeat_interleave_kernel(
    const Tensor& repeats,
    std::optional<int64_t> output_size);

} // namespace at::native::xpu
