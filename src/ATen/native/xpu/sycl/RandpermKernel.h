/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor
randperm_kernel(Tensor& result, int64_t n, std::optional<Generator> generator);

} // namespace at::native::xpu
