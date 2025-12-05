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

TORCH_XPU_API void cat_out_kernel(
    const ITensorListRef& tensors,
    int64_t dim,
    int64_t valid,
    bool all_contiguous,
    bool all_same_dtype,
    bool all_same_sizes_and_stride,
    MemoryFormat memory_format,
    const Tensor& result);

} // namespace at::native::xpu
