/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from FBGEMM
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void permute_2D_lengths_kernel_xpu(
    int32_t T,
    int32_t B,
    const at::Tensor& lengths_contig,
    const at::Tensor& permute_contig,
    at::Tensor& permuted_lengths);

TORCH_XPU_API void permute_2D_data_kernel_xpu(
    int32_t permuted_indices_size,
    int32_t T,
    int32_t B,
    const Tensor& indices_contig,
    const std::optional<const Tensor>& weights,
    const int32_t weights_columns,
    const Tensor& permute_contig,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    Tensor& permuted_indices,
    const std::optional<Tensor>& permuted_weights);

} // namespace at::native::xpu
