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

#include <ATen/ATen.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

namespace sycltla {

bool is_flash_attention_available();

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_flash_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& cumulative_sequence_length_q,
    const std::optional<at::Tensor>& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<at::Tensor>& _seqused_k,
    const std::optional<at::Tensor>& _alibi_slopes,
    const std::optional<at::Tensor>& _block_table,
    std::optional<at::Tensor> out,
    std::optional<int64_t> num_splits);

std::tuple<at::Tensor, at::Tensor, at::Tensor> _flash_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right);

// Deprecated: Use _flash_attention_forward instead.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor>
flash_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const double dropout,
    const bool is_causal,
    const float scale);

// Deprecated: Use _flash_attention_backward instead.
std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    const double dropout,
    const bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const float scale);

} // namespace sycltla
