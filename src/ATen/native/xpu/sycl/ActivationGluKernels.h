/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
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
