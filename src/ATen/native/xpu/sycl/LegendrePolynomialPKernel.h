/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void legendre_polynomial_p_kernel(TensorIteratorBase& iterator);

} // namespace at::native::xpu
