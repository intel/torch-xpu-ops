/*
 * Copyright 2020-2026 Intel Corporation
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
    std::optional<int64_t> num_splits) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "_flash_attention_forward: Torch XPU was not compiled with SYCLTLA support.");
  return std::make_tuple(
      at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor());
#else
  TORCH_CHECK(
      !num_splits.has_value(),
      "num_splits requires FA3. Register FA3 with `register_flash_attention_fa3()` to set num_splits.");

  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  std::optional<at::Tensor> seqused_k = _seqused_k;
  std::optional<at::Tensor> block_table = _block_table;
  std::optional<at::Tensor> alibi_slopes = _alibi_slopes;
  const float softcap = 0.0;
  const int window_left = window_size_left.value_or(-1);
  const int window_right = window_size_right.value_or(-1);
  static_cast<void>(seqused_k);
  static_cast<void>(block_table);

  // We are going to have two paths:
  // 1. The standard MHA path for dense tensors
  // 2. The Varseqlen path
  TORCH_CHECK(
      cumulative_sequence_length_q.has_value() ==
          cumulative_sequence_length_k.has_value(),
      "cumulative_sequence_length_q and cumulative_sequence_length_k must be both set or both not set");

  at::Tensor output, q_padded, k_padded, v_padded, logsumexp, philox_seed,
      philox_offset, debug_attn_mask;
  if (cumulative_sequence_length_q.has_value()) {
    // std::tie(
    //     output,
    //     q_padded,
    //     k_padded,
    //     v_padded,
    //     logsumexp,
    //     philox_seed,
    //     philox_offset,
    //     debug_attn_mask) =
    //     FLASH_NAMESPACE::mha_varlen_fwd(
    //         query,
    //         key,
    //         value,
    //         out,
    //         cumulative_sequence_length_q.value(),
    //         cumulative_sequence_length_k.value(),
    //         seqused_k, /*seqused_k*/
    //         block_table, /*block_table*/
    //         alibi_slopes, /*alibi_slopes*/
    //         max_seqlen_batch_q,
    //         max_seqlen_batch_k,
    //         dropout_p,
    //         softmax_scale,
    //         false /*zero_tensors*/,
    //         is_causal,
    //         window_left,
    //         window_right,
    //         softcap,
    //         return_debug_mask,
    //         std::nullopt /*gen_*/);
    TORCH_CHECK(
        false,
        "_flash_attention_forward: Varseqlen path is not implemented yet.");
  } else {
    std::tie(
        output,
        q_padded,
        k_padded,
        v_padded,
        logsumexp,
        philox_seed,
        philox_offset,
        debug_attn_mask) =
        mha_fwd(
            query,
            key,
            value,
            out,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            is_causal,
            window_left,
            window_right,
            softcap,
            return_debug_mask, /*return_softmax (this is used for testing)*/
            std::nullopt);
    static_cast<void>(q_padded);
    static_cast<void>(k_padded);
    static_cast<void>(v_padded);
  }
  debug_attn_mask =
      return_debug_mask ? debug_attn_mask : at::empty({0}, query.options());
  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));
#endif
}

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
    std::optional<int64_t> window_size_right) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "_flash_attention_backward: Torch XPU was not compiled with SYCLTLA support.");
  return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
#else
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  //  XPU code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();

  const int window_left = window_size_left.value_or(-1);
  const int window_right = window_size_right.value_or(-1);

  std::optional<at::Tensor> dq{std::nullopt};
  std::optional<at::Tensor> dk{std::nullopt};
  std::optional<at::Tensor> dv{std::nullopt};

  // Currently unused args:
  std::optional<at::Tensor> alibi_slopes{std::nullopt};
  const float softcap = 0.0;

  bool deterministic{false};
  auto& ctx = at::globalContext();
  if (ctx.deterministicAlgorithms()) {
    if (ctx.deterministicAlgorithmsWarnOnly()) {
      TORCH_WARN_ONCE(
          "Flash Attention defaults to a non-deterministic algorithm. ",
          "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
    } else {
      deterministic = true;
    }
  }

  // We check the whether the cumulative_sequence_length_q is defined
  // in order to determine whether we are using varlen or dense backward
  if (cumulative_sequence_length_q.defined()) {
    // auto [dQuery, dKey, dValue, dSoftmax] = mha_varlen_bwd(
    //     contiguous_grad_out,
    //     query,
    //     key,
    //     value,
    //     out,
    //     logsumexp,
    //     dq,
    //     dk,
    //     dv,
    //     cumulative_sequence_length_q,
    //     cumulative_sequence_length_k,
    //     alibi_slopes,
    //     max_seqlen_batch_q,
    //     max_seqlen_batch_k,
    //     dropout_p,
    //     softmax_scale,
    //     false /*zero_tensors*/,
    //     is_causal,
    //     window_left,
    //     window_right,
    //     softcap,
    //     deterministic,
    //     philox_seed,
    //     philox_offset);
    // return std::make_tuple(std::move(dQuery), std::move(dKey),
    // std::move(dValue));
    TORCH_CHECK(
        false,
        "_flash_attention_backward: Varseqlen path is not implemented yet.");
  } else {
    auto [dQuery, dKey, dValue, dSoftmax] = mha_bwd(
        contiguous_grad_out,
        query,
        key,
        value,
        out,
        logsumexp,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        is_causal,
        window_left,
        window_right,
        softcap,
        deterministic,
        philox_seed,
        philox_offset);
    return std::make_tuple(
        std::move(dQuery), std::move(dKey), std::move(dValue));
  }
#endif
}

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
    const float scale) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "flash_attention_backward: Torch XPU was not compiled with SYCLTLA support.");
  return std::make_tuple(at::Tensor(), at::Tensor(), at::Tensor());
#else
  //  XPU code assumes that dout is contiguous
  auto contiguous_grad_out = grad_out.contiguous();
  auto [grad_query, grad_key, grad_value] = flash_attention_backward_sycltla(
      contiguous_grad_out,
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
