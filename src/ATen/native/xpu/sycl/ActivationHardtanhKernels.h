/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API void hardtanh_backward_kernel(
    TensorIterator& iter,
    const Scalar& min,
    const Scalar& max);

} // namespace xpu
} // namespace native
} // namespace at
