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

namespace sycltla {

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
mha_fwd(
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        out_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>&
        alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mha_bwd(
    const at::Tensor& dout, // batch_size x seqlen_q x num_heads, x head_size_og
    const at::Tensor& q, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v, // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& out, // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& softmax_lse, // b x h x seqlen_q
    std::optional<at::Tensor>&
        dq_, // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>&
        dk_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        dv_, // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>&
        alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout, // probability to drop
    const float softmax_scale,
    const bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool deterministic,
    const at::Tensor philox_seed,
    const at::Tensor philox_offset);

// Deprecated: Use mha_fwd instead.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    c10::SymInt,
    c10::SymInt,
    at::Tensor,
    at::Tensor>
flash_attention_forward_sycltla(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const double dropout,
    const bool is_causal,
    const float scale);

// Deprecated: Use mha_bwd instead.
std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_backward_sycltla(
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
