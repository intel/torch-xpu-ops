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

  TORCH_XPU_API Tensor split_embedding_codegen_lookup_rowwise_adagrad_function_pt2_xpu(
    const Tensor& placeholder_autograd_tensor,
    const at::TensorList weights,
    const Tensor& D_offsets,
    const c10::SymInt total_D,
    const c10::SymInt max_D,
    const Tensor& hash_size_cumsum,
    const int64_t total_hash_size_bits,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad,
    const int64_t output_dtype,
    const std::vector<std::optional<at::Tensor>>& aux_tensor,
    const std::vector<int64_t>& aux_int,
    const std::vector<double>& aux_float,
    c10::List<bool> aux_bool,
    at::TensorList momentum1, 
    Tensor learning_rate_tensor, 
    std::vector<int64_t> optim_int, 
    std::vector<double> optim_float,
    const c10::SymInt max_B = -1,
    const c10::SymInt max_B_feature_rank = -1,
    const c10::SymInt vbe_output_size = -1
  );

} // namespace at::native::xpu
