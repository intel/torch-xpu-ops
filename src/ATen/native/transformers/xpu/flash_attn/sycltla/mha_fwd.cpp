/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <sycltla/mha_common.h>
#include <sycltla/mha_fwd.h>

namespace cute {

// Declared but not defined here -- explicit specializations are provided by
// the per-headdim compilation units (mha_fwd_hdim*.cpp).
template <typename T, int Headdim, bool IS_CAUSAL>
void run_mha_fwd_(sycl::queue& queue, FLASH_FWD_params& params);

template <typename T, bool IS_CAUSAL>
void run_mha_fwd_(sycl::queue& queue, FLASH_FWD_params& params) {
  const int headdim = params.head_size_vo;

  if (headdim <= 32) {
    run_mha_fwd_<T, 32, IS_CAUSAL>(queue, params);
  } else if (headdim <= 64) {
    run_mha_fwd_<T, 64, IS_CAUSAL>(queue, params);
  } else if (headdim <= 96) {
    run_mha_fwd_<T, 96, IS_CAUSAL>(queue, params);
  } else if (headdim <= 128) {
    run_mha_fwd_<T, 128, IS_CAUSAL>(queue, params);
  } else if (headdim <= 192) {
    run_mha_fwd_<T, 192, IS_CAUSAL>(queue, params);
  } else if (headdim <= 256) {
    run_mha_fwd_<T, 256, IS_CAUSAL>(queue, params);
  } else {
    TORCH_CHECK(
        false,
        "FlashAttentionForwardXPU only support headdim up to 256, got ",
        headdim);
  }
}

void run_mha_fwd(sycl::queue& queue, FLASH_FWD_params& params) {
  FP16_SWITCH(params.is_fp16, [&] {
    BOOL_SWITCH(params.is_causal, IS_CAUSAL, [&] {
      run_mha_fwd_<elem_type, IS_CAUSAL>(queue, params);
    });
  });
}
} // namespace cute

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
    std::optional<at::Generator> gen_) {
  TORCH_CHECK(
      p_dropout == 0.0,
      "mha_fwd on xpu does not only support p_dropout > 0.0 yet");
  TORCH_CHECK(
      alibi_slopes_.has_value() == false,
      "mha_fwd on xpu does not support alibi_slopes yet");
  TORCH_CHECK(
      window_size_left == -1 && window_size_right == -1,
      "mha_fwd on xpu does not support window_SIZE yet");
  TORCH_CHECK(softcap == 0.0, "mha_fwd on xpu does not support softcap yet");
  TORCH_CHECK(
      return_softmax == false,
      "mha_fwd on xpu does not support return_softmax yet");
  TORCH_CHECK(
      !gen_.has_value(),
      "mha_fwd on xpu does not support custom generator yet");

  auto sycl_queue = at::xpu::getCurrentXPUStream().queue();
  auto device_architecture =
      sycl_queue.get_device()
          .get_info<
              sycl::ext::oneapi::experimental::info::device::architecture>();
  constexpr auto supported_architectures =
      std::array<sycl::ext::oneapi::experimental::architecture, 4>{
          sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc_vg,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g21,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g31};
  if (std::find(
          supported_architectures.begin(),
          supported_architectures.end(),
          device_architecture) == supported_architectures.end()) {
    TORCH_CHECK(
        false,
        "XPU device architecture does not support flash attention. Supported architectures are: intel_gpu_pvc, intel_gpu_pvc_vg, intel_gpu_bmg_g21, intel_gpu_bmg_g31.");
  }

  TORCH_CHECK(
      !q.is_nested() && !k.is_nested() && !v.is_nested(),
      "mha_fwd on xpu only supports dense inputs");

  auto dtype = q.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "mha_fwd on xpu only support fp16 and bf16 data type");
  TORCH_CHECK(
      k.scalar_type() == dtype,
      "mha_fwd on xpu: query and key must have the same dtype");
  TORCH_CHECK(
      v.scalar_type() == dtype,
      "mha_fwd on xpu: query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(
      q.stride(-1) == 1,
      "mha_fwd on xpu: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1,
      "mha_fwd on xpu: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1,
      "mha_fwd on xpu: input tensor must have contiguous last dimension");

  TORCH_CHECK(
      q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
      "mha_fwd on xpu requires query, key, value to be 4 dimensional");

  const int batch_size = q.sizes()[0];
  const int seqlen_qo = q.sizes()[1];
  const int seqlen_kv = k.sizes()[1];
  const int numhead_qo = q.sizes()[2];
  const int numhead_kv = k.sizes()[2];
  const int headsize_qk = q.sizes()[3];
  const int headsize_vo = v.sizes()[3];

  CHECK_SHAPE(q, batch_size, seqlen_qo, numhead_qo, headsize_qk);
  CHECK_SHAPE(k, batch_size, seqlen_kv, numhead_kv, headsize_qk);
  CHECK_SHAPE(v, batch_size, seqlen_kv, numhead_kv, headsize_vo);

  if (batch_size == 0) {
    auto opts = q.options();
    at::Tensor out = at::empty({0, seqlen_qo, numhead_qo, headsize_qk}, opts);
    at::Tensor q_padded =
        at::empty({0, seqlen_qo, numhead_qo, headsize_qk}, opts);
    at::Tensor k_padded =
        at::empty({0, seqlen_kv, numhead_kv, headsize_vo}, opts);
    at::Tensor v_padded =
        at::empty({0, seqlen_kv, numhead_kv, headsize_vo}, opts);
    at::Tensor softmax_lse =
        at::empty({0, numhead_qo, seqlen_qo}, opts.dtype(at::kFloat));
    at::Tensor philox_seed = at::empty({}, at::dtype(at::kLong));
    at::Tensor philox_offset = at::empty({}, at::dtype(at::kLong));
    at::Tensor p = at::empty({0}, opts);
    if (return_softmax) {
      p = at::empty({0, numhead_qo, seqlen_qo, seqlen_kv}, opts);
    }
    return {
        std::move(out),
        std::move(q_padded),
        std::move(k_padded),
        std::move(v_padded),
        std::move(softmax_lse),
        std::move(philox_seed),
        std::move(philox_offset),
        std::move(p)};
  }
  TORCH_CHECK(batch_size > 0, "mha_fwd on xpu: batch size must be positive");
  TORCH_CHECK(
      numhead_qo % numhead_kv == 0,
      "mha_fwd on xpu: numhead_qo must be divisible by numhead_kv");
  TORCH_CHECK(
      headsize_vo == headsize_qk,
      "mha_fwd on xpu: only support headsize_qk equal to headsize_vo");
  TORCH_CHECK(
      headsize_qk <= 256,
      "mha_fwd on xpu: only support head dimension at most 256");

  const int headsize_padded = round_up_headdim(headsize_qk);
  const bool needs_headsize_pad = headsize_padded > headsize_qk;

  at::Tensor q_padded = q;
  at::Tensor k_padded = k;
  at::Tensor v_padded = v;
  if (needs_headsize_pad) {
    int pad = headsize_padded - headsize_qk;
    q_padded = at::constant_pad_nd(q, {0, pad}, 0);
    k_padded = at::constant_pad_nd(k, {0, pad}, 0);
    v_padded = at::constant_pad_nd(v, {0, pad}, 0);
  }

  // Base pointers must be 64-byte aligned for block 2D load
  q_padded = ensure_alignment_for_sdpa(q_padded);
  k_padded = ensure_alignment_for_sdpa(k_padded);
  v_padded = ensure_alignment_for_sdpa(v_padded);

  auto opts = q.options();
  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(
        out.dtype() == dtype, "Output must have the same dtype as inputs");
    CHECK_DEVICE(out);
    TORCH_CHECK(
        out.stride(-1) == 1,
        "Output tensor must have contiguous last dimension");
    CHECK_SHAPE(out, batch_size, numhead_qo, seqlen_qo, headsize_vo);
  } else {
    out = at::empty_like(q);
  }

  at::Tensor out_padded = needs_headsize_pad
      ? at::empty({batch_size, seqlen_qo, numhead_qo, headsize_padded}, opts)
      : out;

  if (seqlen_qo > seqlen_kv && is_causal) {
    // When seqlen_qo is greater than seqlen_kv and is_causal(lower_right causal
    // mask) is true, some output positions will skip computation for better
    // performance.
    out_padded.zero_();
  }

  at::Tensor logsumexp =
      at::empty({batch_size, numhead_qo, seqlen_qo}, opts.dtype(at::kFloat));

  at::Tensor p = at::empty({0}, opts);

  FLASH_FWD_params params;
  set_params_fprop(
      params,
      batch_size,
      numhead_qo,
      numhead_kv,
      seqlen_qo,
      seqlen_kv,
      headsize_padded,
      headsize_padded,
      0, // unused seqlen_qo_pad
      0, // unused seqlen_kv_pad
      q_padded,
      k_padded,
      v_padded,
      out_padded,
      logsumexp,
      softmax_scale,
      is_causal);

  at::Tensor philox_seed =
      at::empty({}, at::dtype(c10::kLong).device(at::kXPU));
  at::Tensor philox_offset =
      at::empty({}, at::dtype(c10::kLong).device(at::kXPU));

  if (seqlen_kv > 0) {
    cute::run_mha_fwd(sycl_queue, params);
  } else {
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    out_padded.zero_();
    logsumexp.fill_(-std::numeric_limits<float>::infinity());
  }

  if (needs_headsize_pad) {
    out.copy_(out_padded.slice(/*dim=*/3, /*start=*/0, /*end=*/headsize_qk));
  }

  return {
      std::move(out),
      std::move(q_padded),
      std::move(k_padded),
      std::move(v_padded),
      std::move(logsumexp),
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(p)};
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
flash_attention_forward_sycltla(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const double dropout,
    const bool is_causal,
    const float scale) {
  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  const int seqlen_qo = query.sizes()[2];
  const int seqlen_kv = key.sizes()[2];

  // Query -> Query(Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // Key   -> Key  (Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x Num_heads x Dim_per_head)
  at::Tensor q_t = query.transpose(1, 2);
  at::Tensor k_t = key.transpose(1, 2);
  at::Tensor v_t = value.transpose(1, 2);

  std::optional<at::Tensor> _out = std::nullopt;
  std::optional<at::Tensor> _alibi_slopes = std::nullopt;

  auto
      [out,
       q_padded,
       k_padded,
       v_padded,
       logsumexp,
       philox_seed,
       philox_offset,
       debug_attn_mask] =
          mha_fwd(
              q_t,
              k_t,
              v_t,
              _out,
              _alibi_slopes,
              dropout,
              scale,
              is_causal,
              -1,
              -1,
              0.0,
              false,
              std::nullopt);

  // Reshape output to convert nnz to batch_size and seq_len
  at::Tensor attention = out.transpose(1, 2);
  return {
      std::move(attention),
      std::move(logsumexp),
      /* cumulative_sequence_length_q */ at::Tensor(),
      /* cumulative_sequence_length_k */ at::Tensor(),
      /* max_seqlen_batch_q */ c10::SymInt(seqlen_qo),
      /* max_seqlen_batch_k */ c10::SymInt(seqlen_kv),
      std::move(philox_seed),
      std::move(philox_offset)};
}

} // namespace sycltla
