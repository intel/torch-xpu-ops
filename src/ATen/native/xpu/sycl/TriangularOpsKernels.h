/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void tril_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t k);

TORCH_XPU_API void triu_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t k);

} // namespace at::native::xpu
