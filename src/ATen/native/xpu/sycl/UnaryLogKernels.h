/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void log_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void log10_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void log1p_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void log2_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
