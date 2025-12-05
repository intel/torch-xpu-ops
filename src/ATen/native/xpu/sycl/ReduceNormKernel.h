/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

namespace at::native::xpu {

TORCH_XPU_API void norm_kernel(TensorIterator& iter, const Scalar& val);

} // namespace at::native::xpu
