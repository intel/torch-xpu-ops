/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>
#include <ATen/native/transformers/xpu/flash_attn/sycltla/flash_api.h>

namespace sycltla {

bool is_flash_attention_available() {
#ifndef USE_SYCLTLA
  return false;
#else
  return true;
#endif
}

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
    const float scale) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "flash_attention_forward: Torch XPU was not compiled with SYCLTLA support.");
  return std::make_tuple(
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      c10::SymInt(0),
      c10::SymInt(0),
      at::Tensor(),
      at::Tensor());
#else
  auto
      [attention,
       logsumexp,
       cumulative_sequence_length_q,
       cumulative_sequence_length_k,
       max_seqlen_batch_q,
       max_seqlen_batch_k,
       philox_seed,
       philox_offset] =
          flash_attention_forward_sycltla(
              query, key, value, dropout, is_causal, scale);
  return std::make_tuple(
      std::move(attention),
      std::move(logsumexp),
      std::move(cumulative_sequence_length_q),
      std::move(cumulative_sequence_length_k),
      std::move(max_seqlen_batch_q),
      std::move(max_seqlen_batch_k),
      std::move(philox_seed),
      std::move(philox_offset));
#endif
}

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
    const float scale) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "flash_attention_backward: Torch XPU was not compiled with SYCLTLA support.");
  return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
#else
  auto [grad_query, grad_key, grad_value] = flash_attention_backward_sycltla(
      grad_out,
      query,
      key,
      value,
      out,
      logsumexp,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout,
      is_causal,
      philox_seed,
      philox_offset,
      scale);
  return std::make_tuple(
      std::move(grad_query), std::move(grad_key), std::move(grad_value));
#endif
}
} // namespace sycltla
