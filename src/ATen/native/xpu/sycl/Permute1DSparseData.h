/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * BSD License
 * 
 * For FBGEMM software
 * 
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 *  * Neither the name Facebook nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
