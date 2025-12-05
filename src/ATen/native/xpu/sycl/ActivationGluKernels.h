/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void glu_kernel(TensorIteratorBase& iter);
TORCH_XPU_API void glu_jvp_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void glu_backward_kernel(
    const TensorIteratorBase& iter,
    int64_t gI_stride,
    int64_t I_stride);

} // namespace at::native::xpu
