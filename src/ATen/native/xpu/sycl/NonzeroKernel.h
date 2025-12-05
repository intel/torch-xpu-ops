/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void nonzero_kernel(const Tensor& self, Tensor& out);

} // namespace at::native::xpu
