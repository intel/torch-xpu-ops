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
    const at::Tensor philox_offset) {
  TORCH_CHECK(
      p_dropout == 0.0,
      "mha_bwd on xpu does not only support dropout > 0.0 yet");
  TORCH_CHECK(
      !alibi_slopes_.has_value(),
      "mha_bwd on xpu does not support alibi slopes yet");
  TORCH_CHECK(
      window_size_left == -1 && window_size_right == -1,
      "mha_bwd on xpu does not support windowed attention yet");
  TORCH_CHECK(softcap == 0.0, "mha_bwd on xpu does not support softcap yet");
  TORCH_CHECK(
      !deterministic, "mha_bwd on xpu does not support deterministic mode yet");

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

  TORCH_CHECK(
      !q.is_nested() && !k.is_nested() && !v.is_nested() && !out.is_nested() &&
          !dout.is_nested() && !softmax_lse.is_nested(),
      "mha_bwd on xpu only support dense inputs");

  auto dtype = q.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "mha_bwd on xpu only support fp16 and bf16 data type");
  TORCH_CHECK(
      softmax_lse.scalar_type() == at::kFloat,
      "mha_bwd on xpu: logsumexp must have the dtype float32");
  TORCH_CHECK(
      k.scalar_type() == dtype,
      "mha_bwd on xpu: query and key must have the same dtype");
  TORCH_CHECK(
      v.scalar_type() == dtype,
      "mha_bwd on xpu: query and value must have the same dtype");
  TORCH_CHECK(
      out.scalar_type() == dtype,
      "mha_bwd on xpu: query and out must have the same dtype");
  TORCH_CHECK(
      dout.scalar_type() == dtype,
      "mha_bwd on xpu: query and grad_out must have the same dtype");

  TORCH_CHECK(
      q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4 &&
          dout.dim() == 4 && softmax_lse.dim() == 3,
      "mha_bwd on xpu requires query, key, value, out, grad_out to be 4 dimensional and logsumexp to be 3 dimensional");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(softmax_lse);

  TORCH_CHECK(
      q.stride(-1) == 1,
      "mha_bwd on xpu: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1,
      "mha_bwd on xpu: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1,
      "mha_bwd on xpu: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      out.stride(-1) == 1,
      "mha_bwd on xpu: out tensor must have contiguous last dimension");
  TORCH_CHECK(
      dout.stride(-1) == 1,
      "mha_bwd on xpu: dout tensor must have contiguous last dimension");
  TORCH_CHECK(
      softmax_lse.stride(-1) == 1,
      "mha_bwd on xpu: softmax_lse tensor must have contiguous last dimension");
  TORCH_CHECK(
      softmax_lse.is_contiguous(),
      "mha_bwd on xpu: softmax_lse must be contiguous in [batch_size, numhead_qo, seqlen_qo]");

  const int batch_size = q.sizes()[0];
  const int seqlen_qo = q.sizes()[1];
  const int seqlen_kv = k.sizes()[1];
  const int numhead_qo = q.sizes()[2];
  const int numhead_kv = k.sizes()[2];
  const int headsize_qk = q.sizes()[3];
  const int headsize_vo = v.sizes()[3];

  TORCH_CHECK(
      headsize_qk == headsize_vo,
      "mha_bwd on xpu: headsize_qk must be equal to headsize_vo");

  if (batch_size == 0) {
    auto opts = q.options();
    at::Tensor dq = at::empty_like(q);
    at::Tensor dk = at::empty_like(k);
    at::Tensor dv = at::empty_like(v);
    at::Tensor softmax_d =
        at::empty({0, numhead_qo, seqlen_qo}, opts.dtype(at::kFloat));
    return {dq, dk, dv, softmax_d};
  }
  TORCH_CHECK(batch_size > 0, "mha_bwd on xpu: batch size must be positive");
  TORCH_CHECK(
      headsize_qk <= 256,
      "mha_bwd on xpu: only supports head dimension at most 256");
  TORCH_CHECK(
      numhead_qo % numhead_kv == 0,
      "mha_bwd on xpu: number of heads in key/value must divide number of heads in query");

  CHECK_SHAPE(q, batch_size, seqlen_qo, numhead_qo, headsize_qk);
  CHECK_SHAPE(k, batch_size, seqlen_kv, numhead_kv, headsize_qk);
  CHECK_SHAPE(v, batch_size, seqlen_kv, numhead_kv, headsize_vo);
  CHECK_SHAPE(out, batch_size, seqlen_qo, numhead_qo, headsize_vo);
  CHECK_SHAPE(dout, batch_size, seqlen_qo, numhead_qo, headsize_vo);
  CHECK_SHAPE(softmax_lse, batch_size, numhead_qo, seqlen_qo);

  at::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == dtype, "dq must have the same dtype as q");
    CHECK_DEVICE(dq);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    CHECK_SHAPE(dq, batch_size, seqlen_qo, numhead_qo, headsize_qk);
  } else {
    dq = at::empty_like(q);
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == dtype, "dk must have the same dtype as q");
    CHECK_DEVICE(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    CHECK_SHAPE(dk, batch_size, seqlen_kv, numhead_kv, headsize_qk);
  } else {
    dk = at::empty_like(k);
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == dtype, "dv must have the same dtype as q");
    CHECK_DEVICE(dv);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    CHECK_SHAPE(dv, batch_size, seqlen_kv, numhead_kv, headsize_vo);
  } else {
    dv = at::empty_like(v);
  }

  auto softmax_d = at::Tensor();

  const int headsize_padded = round_up_headdim(headsize_qk);
  const bool needs_headsize_pad = headsize_padded > headsize_qk;

  at::Tensor q_padded = q;
  at::Tensor k_padded = k;
  at::Tensor v_padded = v;
  at::Tensor out_padded = out;
  at::Tensor dout_padded = dout;
  if (needs_headsize_pad) {
    int pad = headsize_padded - headsize_qk;
    q_padded = at::constant_pad_nd(q, {0, pad}, 0);
    k_padded = at::constant_pad_nd(k, {0, pad}, 0);
    v_padded = at::constant_pad_nd(v, {0, pad}, 0);
    out_padded = at::constant_pad_nd(out, {0, pad}, 0);
    dout_padded = at::constant_pad_nd(dout, {0, pad}, 0);
  }

  q_padded = ensure_alignment_for_sdpa(q_padded);
  k_padded = ensure_alignment_for_sdpa(k_padded);
  v_padded = ensure_alignment_for_sdpa(v_padded);
  out_padded = ensure_alignment_for_sdpa(out_padded);
  dout_padded = ensure_alignment_for_sdpa(dout_padded);

  auto opts = q.options();
  // Allocate padded buffers for the kernel when headsize padding is needed.
  auto dq_padded = needs_headsize_pad
      ? at::empty({batch_size, seqlen_qo, numhead_qo, headsize_padded}, opts)
      : dq;
  auto dk_padded = needs_headsize_pad
      ? at::empty({batch_size, seqlen_kv, numhead_kv, headsize_padded}, opts)
      : dk;
  auto dv_padded = needs_headsize_pad
      ? at::empty({batch_size, seqlen_kv, numhead_kv, headsize_padded}, opts)
      : dv;

  const bool is_gqa = (numhead_kv != numhead_qo);
  at::Tensor dk_expanded, dv_expanded;
  if (is_gqa) {
    dk_expanded =
        at::empty({batch_size, seqlen_kv, numhead_qo, headsize_padded}, opts);
    dv_expanded =
        at::empty({batch_size, seqlen_kv, numhead_qo, headsize_padded}, opts);
  } else {
    dk_expanded = dk_padded;
    dv_expanded = dv_padded;
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
      q_padded,
      k_padded,
      v_padded,
      out_padded,
      dout_padded,
      softmax_lse,
      dq_padded,
      dk_expanded,
      dv_expanded,
      tensor_odo,
      tensor_dqaccum,
      tensor_pbuff,
      softmax_scale,
      is_causal);

  if (seqlen_qo > 0) {
    cute::run_mha_bwd(sycl_queue, params);
  } else {
    dk_expanded.zero_();
    dv_expanded.zero_();
  }

  if (is_gqa) {
    if (needs_headsize_pad) {
      auto dk_reduced =
          at::empty({batch_size, seqlen_kv, numhead_kv, headsize_padded}, opts);
      auto dv_reduced =
          at::empty({batch_size, seqlen_kv, numhead_kv, headsize_padded}, opts);
      at::sum_out(
          dk_reduced,
          at::reshape(
              dk_expanded,
              {batch_size,
               seqlen_kv,
               numhead_kv,
               numhead_qo / numhead_kv,
               headsize_padded}),
          {3});
      at::sum_out(
          dv_reduced,
          at::reshape(
              dv_expanded,
              {batch_size,
               seqlen_kv,
               numhead_kv,
               numhead_qo / numhead_kv,
               headsize_padded}),
          {3});
      dk_padded = dk_reduced;
      dv_padded = dv_reduced;
    } else {
      at::sum_out(
          dk,
          at::reshape(
              dk_expanded,
              {batch_size,
               seqlen_kv,
               numhead_kv,
               numhead_qo / numhead_kv,
               headsize_padded}),
          {3});
      at::sum_out(
          dv,
          at::reshape(
              dv_expanded,
              {batch_size,
               seqlen_kv,
               numhead_kv,
               numhead_qo / numhead_kv,
               headsize_padded}),
          {3});
      at::sum_out(
          dv,
          at::reshape(
              dv_expanded,
              {batch_size,
               seqlen_kv,
               numhead_kv,
               numhead_qo / numhead_kv,
               headsize_padded}),
          {3});
    }
  }

  if (needs_headsize_pad) {
    dq.copy_(dq_padded.slice(/*dim=*/3, /*start=*/0, /*end=*/headsize_qk));
    dk.copy_(dk_padded.slice(/*dim=*/3, /*start=*/0, /*end=*/headsize_qk));
    dv.copy_(dv_padded.slice(/*dim=*/3, /*start=*/0, /*end=*/headsize_vo));
  }

  return std::make_tuple(
      std::move(dq), std::move(dk), std::move(dv), std::move(softmax_d));
}

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
  at::Tensor q_t = query.transpose(1, 2);
  at::Tensor k_t = key.transpose(1, 2);
  at::Tensor v_t = value.transpose(1, 2);

  at::Tensor grad_out_t = grad_out.transpose(1, 2);
  at::Tensor out_t = out.transpose(1, 2);

  std::optional<at::Tensor> dq = std::nullopt, dk = std::nullopt,
                            dv = std::nullopt, alibi_slopes = std::nullopt;
  auto [grad_q, grad_k, grad_v, softmax_d] = mha_bwd(
      grad_out_t,
      q_t,
      k_t,
      v_t,
      out_t,
      logsumexp,
      dq,
      dk,
      dv,
      alibi_slopes,
      dropout,
      scale,
      is_causal,
      -1,
      -1,
      0.0,
      false,
      philox_seed,
      philox_offset);

  grad_q = grad_q.transpose(1, 2);
  grad_k = grad_k.transpose(1, 2);
  grad_v = grad_v.transpose(1, 2);

  return std::make_tuple(grad_q, grad_k, grad_v);
}
} // namespace sycltla
