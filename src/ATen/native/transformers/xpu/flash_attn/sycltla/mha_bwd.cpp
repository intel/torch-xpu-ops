/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <sycltla/mha_bwd.h>
#include <sycltla/mha_common.h>

namespace cute {

// Declared but not defined here -- explicit specializations are provided by
// the per-headdim compilation units (mha_bwd_hdim*.cpp).
template <typename T, int Headdim, bool is_causal>
void run_mha_bwd_(sycl::queue& queue, FLASH_BWD_params& params);

template <typename T, bool is_causal>
void run_mha_bwd_(sycl::queue& queue, FLASH_BWD_params& params) {
  const int headdim = params.head_size_vo;

  if (headdim <= 32) {
    run_mha_bwd_<T, 32, is_causal>(queue, params);
  } else if (headdim <= 64) {
    run_mha_bwd_<T, 64, is_causal>(queue, params);
  } else if (headdim <= 96) {
    run_mha_bwd_<T, 96, is_causal>(queue, params);
  } else if (headdim <= 128) {
    run_mha_bwd_<T, 128, is_causal>(queue, params);
  } else if (headdim <= 192) {
    run_mha_bwd_<T, 192, is_causal>(queue, params);
  } else if (headdim <= 256) {
    run_mha_bwd_<T, 256, is_causal>(queue, params);
  } else {
    TORCH_CHECK(
        false,
        "FlashAttentionBackwardXPU only support headdim up to 256, got ",
        headdim);
  }
}

void run_mha_bwd(sycl::queue& queue, FLASH_BWD_params& params) {
  FP16_SWITCH(params.is_fp16, [&] {
    BOOL_SWITCH(params.is_causal, IS_CAUSAL, [&] {
      run_mha_bwd_<elem_type, IS_CAUSAL>(queue, params);
    });
  });
}

} // namespace cute

namespace sycltla {

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
    const float scale) {
  TORCH_CHECK(
      dropout == 0.0,
      "FlashAttentionBackwardXPU does not only support dropout > 0.0 yet");

  at::Tensor contiguous_grad_out = grad_out.contiguous();

  CHECK_DEVICE(query);
  CHECK_DEVICE(key);
  CHECK_DEVICE(value);
  CHECK_DEVICE(out);
  CHECK_DEVICE(contiguous_grad_out);
  CHECK_DEVICE(logsumexp);

  TORCH_CHECK(
      !query.is_nested() && !key.is_nested() && !value.is_nested() &&
          !out.is_nested() && !grad_out.is_nested() && !logsumexp.is_nested(),
      "FlashAttentionBackwardXPU only support dense inputs");

  auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "FlashAttentionBackwardXPU only support fp16 and bf16 data type");
  TORCH_CHECK(
      logsumexp.scalar_type() == at::kFloat,
      "FlashAttentionBackwardXPU: logsumexp must have the dtype float32");
  TORCH_CHECK(
      key.scalar_type() == dtype,
      "FlashAttentionBackwardXPU: query and key must have the same dtype");
  TORCH_CHECK(
      value.scalar_type() == dtype,
      "FlashAttentionBackwardXPU: query and value must have the same dtype");
  TORCH_CHECK(
      out.scalar_type() == dtype,
      "FlashAttentionBackwardXPU: query and out must have the same dtype");
  TORCH_CHECK(
      contiguous_grad_out.scalar_type() == dtype,
      "FlashAttentionBackwardXPU: query and grad_out must have the same dtype");

  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
          out.dim() == 4 && contiguous_grad_out.dim() == 4 &&
          logsumexp.dim() == 3,
      "FlashAttentionBackwardXPU requires query, key, value, out, grad_out to be 4 dimensional and logsumexp to be 3 dimensional");

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
  CHECK_SHAPE(out, batch_size, numhead_qo, seqlen_qo, headsize_vo);
  CHECK_SHAPE(
      contiguous_grad_out, batch_size, numhead_qo, seqlen_qo, headsize_vo);
  CHECK_SHAPE(logsumexp, batch_size, numhead_qo, seqlen_qo);
  TORCH_CHECK(
      numhead_qo % numhead_kv == 0,
      "FlashAttentionBackwardXPU: number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(
      headsize_qk == headsize_vo,
      "FlashAttentionBackwardXPU: headsize_qk must be equal to headsize_vo");

  TORCH_CHECK(
      query.stride(-1) == 1,
      "FlashAttentionBackwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      key.stride(-1) == 1,
      "FlashAttentionBackwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      value.stride(-1) == 1,
      "FlashAttentionBackwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      out.stride(-1) == 1,
      "FlashAttentionBackwardXPU: out tensor must have contiguous last dimension");
  TORCH_CHECK(
      contiguous_grad_out.stride(-1) == 1,
      "FlashAttentionBackwardXPU: dout tensor must have contiguous last dimension");
  TORCH_CHECK(
      logsumexp.stride(-1) == 1,
      "FlashAttentionBackwardXPU: logsumexp tensor must have contiguous last dimension");
  TORCH_CHECK(
      logsumexp.is_contiguous(),
      "FlashAttentionBackwardXPU: logsumexp must be contiguous in [batch_size, numhead_qo, seqlen_qo]");

  const int headsize_padded = round_up_headdim(headsize_qk);
  const bool needs_headsize_pad = headsize_padded > headsize_qk;

  at::Tensor q_padded = query;
  at::Tensor k_padded = key;
  at::Tensor v_padded = value;
  at::Tensor out_padded = out;
  at::Tensor grad_out_padded = contiguous_grad_out;
  if (needs_headsize_pad) {
    int pad = headsize_padded - headsize_qk;
    q_padded = at::constant_pad_nd(query, {0, pad}, 0);
    k_padded = at::constant_pad_nd(key, {0, pad}, 0);
    v_padded = at::constant_pad_nd(value, {0, pad}, 0);
    out_padded = at::constant_pad_nd(out, {0, pad}, 0);
    grad_out_padded = at::constant_pad_nd(contiguous_grad_out, {0, pad}, 0);
  }

  at::Tensor q_aligned = ensure_alignment_for_sdpa(q_padded);
  at::Tensor k_aligned = ensure_alignment_for_sdpa(k_padded);
  at::Tensor v_aligned = ensure_alignment_for_sdpa(v_padded);
  at::Tensor out_aligned = ensure_alignment_for_sdpa(out_padded);
  at::Tensor grad_out_aligned = ensure_alignment_for_sdpa(grad_out_padded);

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
        "XPU device architecture does not support flash attention backward. Supported architectures are: intel_gpu_pvc, intel_gpu_pvc_vg, intel_gpu_bmg_g21, intel_gpu_bmg_g31.");
  }

  auto opts = query.options();

  auto grad_query = at::empty_like(query);
  auto grad_key = at::empty_like(key);
  auto grad_value = at::empty_like(value);

  // Allocate padded buffers for the kernel when headsize padding is needed.
  auto grad_query_padded = needs_headsize_pad
      ? at::empty({batch_size, numhead_qo, seqlen_qo, headsize_padded}, opts)
      : grad_query;
  auto grad_key_padded = needs_headsize_pad
      ? at::empty({batch_size, numhead_kv, seqlen_kv, headsize_padded}, opts)
      : grad_key;
  auto grad_value_padded = needs_headsize_pad
      ? at::empty({batch_size, numhead_kv, seqlen_kv, headsize_padded}, opts)
      : grad_value;

  const bool is_gqa = (numhead_kv != numhead_qo);
  at::Tensor grad_key_expanded, grad_value_expanded;
  if (is_gqa) {
    grad_key_expanded =
        at::empty({batch_size, numhead_qo, seqlen_kv, headsize_padded}, opts);
    grad_value_expanded =
        at::empty({batch_size, numhead_qo, seqlen_kv, headsize_padded}, opts);
  } else {
    grad_key_expanded = grad_key_padded;
    grad_value_expanded = grad_value_padded;
  }

  int seqlen_qo_pad = (seqlen_qo + kBwdMPad - 1) / kBwdMPad * kBwdMPad;
  int seqlen_kv_pad = (seqlen_kv + kBwdNPad - 1) / kBwdNPad * kBwdNPad;
  auto tensor_odo =
      at::empty({batch_size, numhead_qo, seqlen_qo}, opts.dtype(at::kFloat));
  auto tensor_dqaccum = at::zeros(
      {batch_size, numhead_qo, seqlen_qo_pad, headsize_padded},
      opts.dtype(at::kFloat));
  auto tensor_pbuff =
      at::empty({batch_size, numhead_qo, seqlen_kv_pad, 2 * kBwdMPad}, opts);

  FLASH_BWD_params params;
  set_params_dgrad(
      params,
      batch_size,
      numhead_qo,
      numhead_kv,
      seqlen_qo,
      seqlen_kv,
      headsize_padded,
      headsize_padded,
      seqlen_qo_pad,
      seqlen_kv_pad,
      q_aligned,
      k_aligned,
      v_aligned,
      out_aligned,
      grad_out_aligned,
      logsumexp,
      grad_query_padded,
      grad_key_expanded,
      grad_value_expanded,
      tensor_odo,
      tensor_dqaccum,
      tensor_pbuff,
      scale,
      is_causal);

  cute::run_mha_bwd(sycl_queue, params);

  if (is_gqa) {
    if (needs_headsize_pad) {
      auto grad_key_reduced =
          at::empty({batch_size, numhead_kv, seqlen_kv, headsize_padded}, opts);
      auto grad_value_reduced =
          at::empty({batch_size, numhead_kv, seqlen_kv, headsize_padded}, opts);
      at::sum_out(
          grad_key_reduced,
          at::reshape(
              grad_key_expanded,
              {batch_size,
               numhead_kv,
               numhead_qo / numhead_kv,
               seqlen_kv,
               headsize_padded}),
          {2});
      at::sum_out(
          grad_value_reduced,
          at::reshape(
              grad_value_expanded,
              {batch_size,
               numhead_kv,
               numhead_qo / numhead_kv,
               seqlen_kv,
               headsize_padded}),
          {2});
      grad_key_padded = grad_key_reduced;
      grad_value_padded = grad_value_reduced;
    } else {
      at::sum_out(
          grad_key,
          at::reshape(
              grad_key_expanded,
              {batch_size,
               numhead_kv,
               numhead_qo / numhead_kv,
               seqlen_kv,
               headsize_padded}),
          {2});
      at::sum_out(
          grad_value,
          at::reshape(
              grad_value_expanded,
              {batch_size,
               numhead_kv,
               numhead_qo / numhead_kv,
               seqlen_kv,
               headsize_padded}),
          {2});
    }
  }

  if (needs_headsize_pad) {
    grad_query.copy_(
        grad_query_padded.slice(/*dim=*/3, /*start=*/0, /*end=*/headsize_qk));
    grad_key.copy_(
        grad_key_padded.slice(/*dim=*/3, /*start=*/0, /*end=*/headsize_qk));
    grad_value.copy_(
        grad_value_padded.slice(/*dim=*/3, /*start=*/0, /*end=*/headsize_vo));
  }

  return std::make_tuple(
      std::move(grad_query), std::move(grad_key), std::move(grad_value));
}
} // namespace sycltla
