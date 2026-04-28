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

  /**
  * @brief SYCL/XPU implementation of permute_1D_sparse_data operator
  *
  * This operator permutes sparse data represented in jagged/1D format.
  * It reorders (permutes) segments of data according to a permutation array.
  *
  * The operation is commonly used in:
  * - Deep Learning Recommendation Models (DLRMs) for reordering embedding features
  * - Sparse tensor operations in graph neural networks
  * - Distributed training for all-to-all communication preparation
  *
  * Input format (jagged/CSR-like):
  *   lengths: [L0, L1, L2, ...]  - length of each segment
  *   values:  [seg0 data][seg1 data][seg2 data]...  - concatenated values
  *   permute: [P0, P1, P2, ...]  - new segment order
  *
  * Output:
  *   permuted_lengths[j] = lengths[permute[j]]
  *   permuted_values contains data from segment permute[j] at position j
  *
  * @param permute Permutation indices tensor [P] - int32
  * @param lengths Segment lengths tensor [L] - int32/int64
  * @param indices Concatenated values tensor [V] - any type
  * @param weights Optional weights tensor [V] - float/double
  * @param permuted_lengths_sum Optional precomputed sum of permuted lengths
  * @return Tuple of (permuted_lengths, permuted_indices, permuted_weights)
  */
  TORCH_XPU_API std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
  permute_1D_sparse_data_xpu(
      const at::Tensor& permute,
      const at::Tensor& lengths,
      const at::Tensor& indices,
      const std::optional<at::Tensor>& weights,
      const std::optional<int64_t>& permuted_lengths_sum);
} // namespace at::native::xpu
