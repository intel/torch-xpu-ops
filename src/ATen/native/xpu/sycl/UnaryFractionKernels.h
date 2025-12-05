/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void reciprocal_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void floor_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void ceil_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void round_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void round_decimals_kernel(
    TensorIteratorBase& iter,
    int64_t decimals);

TORCH_XPU_API void frac_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void trunc_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
