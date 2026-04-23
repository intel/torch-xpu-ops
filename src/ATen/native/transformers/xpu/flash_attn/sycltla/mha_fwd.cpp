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
#include <sycltla/mha_fwd_launch.h>

namespace cute {

// Explicit specializations for all headdim variants (merged into a single
// compilation unit to reduce peak memory during parallel builds).

#define INSTANTIATE_FWD_HDIM(T, HDIM, CAUSAL, hdim_fn)  \
  template <>                                           \
  void run_mha_fwd_<T, HDIM, CAUSAL>(                   \
      sycl::queue & queue, FLASH_FWD_params & params) { \
    hdim_fn<T, CAUSAL>(queue, params);                  \
  }

#define INSTANTIATE_FWD_HDIM_ALL(HDIM, hdim_fn)                \
  INSTANTIATE_FWD_HDIM(cute::half_t, HDIM, false, hdim_fn)     \
  INSTANTIATE_FWD_HDIM(cute::half_t, HDIM, true, hdim_fn)      \
  INSTANTIATE_FWD_HDIM(cute::bfloat16_t, HDIM, false, hdim_fn) \
  INSTANTIATE_FWD_HDIM(cute::bfloat16_t, HDIM, true, hdim_fn)

INSTANTIATE_FWD_HDIM_ALL(32, run_mha_fwd_hdim32)
INSTANTIATE_FWD_HDIM_ALL(64, run_mha_fwd_hdim64)
INSTANTIATE_FWD_HDIM_ALL(96, run_mha_fwd_hdim96)
INSTANTIATE_FWD_HDIM_ALL(128, run_mha_fwd_hdim128)
INSTANTIATE_FWD_HDIM_ALL(192, run_mha_fwd_hdim192)
INSTANTIATE_FWD_HDIM_ALL(256, run_mha_fwd_hdim256)

#undef INSTANTIATE_FWD_HDIM_ALL
#undef INSTANTIATE_FWD_HDIM

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
  TORCH_CHECK(
      dropout == 0.0,
      "FlashAttentionForwardXPU does not only support dropout > 0.0 yet");

  CHECK_DEVICE(query);
  CHECK_DEVICE(key);
  CHECK_DEVICE(value);

  TORCH_CHECK(
      !query.is_nested() && !key.is_nested() && !value.is_nested(),
      "FlashAttentionForwardXPU only support dense inputs");

  auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "FlashAttentionForwardXPU only support fp16 and bf16 data type");
  TORCH_CHECK(
      key.scalar_type() == dtype,
      "FlashAttentionForwardXPU: query and key must have the same dtype");
  TORCH_CHECK(
      value.scalar_type() == dtype,
      "FlashAttentionForwardXPU: query and value must have the same dtype");

  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "FlashAttentionForwardXPU requires query, key, value to be 4 dimensional");

  const int batch_size = query.sizes()[0];
  const int numhead_qo = query.sizes()[1];
  const int numhead_kv = key.sizes()[1];
  const int seqlen_qo = query.sizes()[2];
  const int seqlen_kv = key.sizes()[2];
  const int headsize_qk = query.sizes()[3];
  const int headsize_vo = value.sizes()[3];

  CHECK_SHAPE(query, batch_size, numhead_qo, seqlen_qo, headsize_qk);
  CHECK_SHAPE(key, batch_size, numhead_kv, seqlen_kv, headsize_qk);
  CHECK_SHAPE(value, batch_size, numhead_kv, seqlen_kv, headsize_vo);

  TORCH_CHECK(
      numhead_qo % numhead_kv == 0,
      "FlashAttentionForwardXPU: numhead_qo must be divisible by numhead_kv");

  TORCH_CHECK(
      query.stride(-1) == 1,
      "FlashAttentionForwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      key.stride(-1) == 1,
      "FlashAttentionForwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      value.stride(-1) == 1,
      "FlashAttentionForwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      headsize_vo == headsize_qk,
      "FlashAttentionForwardXPU only support headsize_qk equal to headsize_vo");

  const int headsize_padded = round_up_headdim(headsize_qk);
  const bool needs_headsize_pad = headsize_padded > headsize_qk;

  at::Tensor q_padded = query;
  at::Tensor k_padded = key;
  at::Tensor v_padded = value;
  if (needs_headsize_pad) {
    int pad = headsize_padded - headsize_qk;
    q_padded = at::constant_pad_nd(query, {0, pad}, 0);
    k_padded = at::constant_pad_nd(key, {0, pad}, 0);
    v_padded = at::constant_pad_nd(value, {0, pad}, 0);
  }

  auto opts = query.options();
  at::Tensor out = at::empty_like(query);

  at::Tensor out_padded = needs_headsize_pad
      ? at::empty({batch_size, numhead_qo, seqlen_qo, headsize_padded}, opts)
      : out;

  if (seqlen_qo > seqlen_kv && is_causal) {
    // When seqlen_qo is greater than seqlen_kv and is_causal(lower_right causal
    // mask) is true, some output positions will skip computation for better
    // performance.
    out_padded.zero_();
  }

  at::Tensor logsumexp =
      at::empty({batch_size, numhead_qo, seqlen_qo}, opts.dtype(at::kFloat));

  // Base pointers must be 64-byte aligned for block 2D load
  at::Tensor q_aligned = ensure_alignment_for_sdpa(q_padded);
  at::Tensor k_aligned = ensure_alignment_for_sdpa(k_padded);
  at::Tensor v_aligned = ensure_alignment_for_sdpa(v_padded);

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
      q_aligned,
      k_aligned,
      v_aligned,
      out_padded,
      logsumexp,
      scale,
      is_causal);

  cute::run_mha_fwd(sycl_queue, params);

  if (needs_headsize_pad) {
    out.copy_(out_padded.slice(/*dim=*/3, /*start=*/0, /*end=*/headsize_qk));
  }

  return std::tuple<
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      c10::SymInt,
      c10::SymInt,
      at::Tensor,
      at::Tensor>{
      out,
      logsumexp,
      /* cumulative_sequence_length_q */ at::Tensor(),
      /* cumulative_sequence_length_k */ at::Tensor(),
      /* max_seqlen_batch_q */ c10::SymInt(seqlen_qo),
      /* max_seqlen_batch_k */ c10::SymInt(seqlen_kv),
      /* philox_seed */ at::empty({}, at::dtype(at::kLong)),
      /* philox_offset */ at::empty({}, at::dtype(at::kLong))};
}

} // namespace sycltla
