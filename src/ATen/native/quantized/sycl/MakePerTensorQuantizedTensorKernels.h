/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void assign_quantized_tensor_kernel(
    const Tensor& self,
    Tensor& dst);

} // namespace at::native::xpu
