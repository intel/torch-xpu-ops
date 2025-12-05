/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void nextafter_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void heaviside_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
