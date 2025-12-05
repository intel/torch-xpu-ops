/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void log_sigmoid_forward_kernel(TensorIteratorBase& iter);

TORCH_XPU_API void log_sigmoid_backward_kernel(TensorIterator& iter);

} // namespace at::native::xpu
