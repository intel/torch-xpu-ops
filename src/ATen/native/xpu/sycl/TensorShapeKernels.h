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

TORCH_XPU_API void split_with_sizes_copy_out_xpu_kernel(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim,
    TensorList out);

TORCH_XPU_API Tensor
_chunk_cat_xpu_kernel(TensorList tensors, int64_t dim, int64_t num_chunks);

TORCH_XPU_API Tensor& _chunk_cat_out_xpu_kernel(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out);

} // namespace at::native::xpu
