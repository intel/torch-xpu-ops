/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void flip_kernel(TensorIterator& iter, bool quantized);

// Rolls `input` along `dims` into the contiguous `output` in a single pass.
// Returns false if the shape is not supported (64-bit indexing).
TORCH_XPU_API bool roll_kernel(
    const Tensor& input,
    Tensor& output,
    IntArrayRef shifts,
    IntArrayRef dims);

} // namespace at::native::xpu
