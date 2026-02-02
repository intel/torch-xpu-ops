/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <cassert>

#include <ATen/xpu/XPUContext.h>

#include <ATen/native/transformers/xpu/flash_attn/flash_api.h>
#include <ATen/native/transformers/xpu/flash_attn/sycltla/flash_api.h>
#include <ATen/native/transformers/xpu/flash_attn/utils.h>

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")

#define CHECK_SHAPE(x, ...)                        \
  TORCH_CHECK(                                     \
      x.sizes() == at::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)            \
  [&] {                                   \
    if (COND) {                           \
      using elem_type = cute::half_t;     \
      return __VA_ARGS__();               \
    } else {                              \
      using elem_type = cute::bfloat16_t; \
      return __VA_ARGS__();               \
    }                                     \
  }()

struct QKV_params {
  using index_t = int64_t;
  // The QKV matrices.
  void* q_ptr;
  void* k_ptr;
  void* v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
};

struct FLASH_FWD_params : public QKV_params {
  // The Output matrix.
  void* o_ptr;

  // The stride between rows of the Output matrix.
  index_t o_batch_stride;
  index_t o_head_stride;
  index_t o_row_stride;

  // The pointer to the softmax logsumexp.
  void* lse_ptr;

  // The dimensions of the problem.
  int batch_size;
  int num_heads_qo;
  int num_heads_kv;
  int seqlen_qo;
  int seqlen_kv;
  int head_size_qk;
  int head_size_vo;
  int seqlen_qo_pad;
  int seqlen_kv_pad;

  bool is_causal;
  float scale;
  bool is_fp16;
};

struct FLASH_BWD_params : public FLASH_FWD_params {
  // The dO and dQKV matrices.
  void* do_ptr;
  void* dq_ptr;
  void* dk_ptr;
  void* dv_ptr;

  // To accumulate dQ.
  void* dqaccum_ptr;

  // To precompute o*do.
  void* odo_ptr;

  // To reorder P and dP due to slm api not ready.
  void* pb_ptr;

  // The stride between rows of the dO, dQ, dK and dV matrices.
  index_t do_batch_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;
  index_t do_head_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;
  index_t do_row_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
};

void set_params_fprop(
    FLASH_FWD_params& params,
    // sizes
    const int batch_size,
    const int num_heads_qo,
    const int num_heads_kv,
    const int seqlen_qo,
    const int seqlen_kv,
    const int head_size_qk,
    const int head_size_vo,
    const int seqlen_qo_pad,
    const int seqlen_kv_pad,
    // device pointers
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    at::Tensor& logsumexp,
    float scale,
    bool is_causal) {
  // Reset the parameters
  params = {};

  params.is_fp16 = q.dtype() == at::kHalf;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();
  params.lse_ptr = logsumexp.data_ptr();
  // All stride are in elements, not bytes.
  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.o_batch_stride = out.stride(0);
  params.q_head_stride = q.stride(1);
  params.k_head_stride = k.stride(1);
  params.v_head_stride = v.stride(1);
  params.o_head_stride = out.stride(1);
  params.q_row_stride = q.stride(2);
  params.k_row_stride = k.stride(2);
  params.v_row_stride = v.stride(2);
  params.o_row_stride = out.stride(2);

  // Set the dimensions.
  params.batch_size = batch_size;
  params.num_heads_qo = num_heads_qo;
  params.num_heads_kv = num_heads_kv;
  params.seqlen_qo = seqlen_qo;
  params.seqlen_kv = seqlen_kv;
  params.head_size_qk = head_size_qk;
  params.head_size_vo = head_size_vo;
  params.seqlen_qo_pad = seqlen_qo_pad;
  params.seqlen_kv_pad = seqlen_kv_pad;

  // Other params
  params.is_causal = is_causal;
  params.scale = scale;
}

void set_params_dgrad(
    FLASH_BWD_params& params,
    // sizes
    const int batch_size,
    const int num_heads_qo,
    const int num_heads_kv,
    const int seqlen_qo,
    const int seqlen_kv,
    const int head_size_qk,
    const int head_size_vo,
    const int seqlen_qo_pad,
    const int seqlen_kv_pad,
    // device pointers
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    // temp buffers
    at::Tensor& tensor_odo,
    at::Tensor& tensor_dqaccum,
    at::Tensor& tensor_pbuff,
    // other params
    float scale,
    bool is_causal) {
  params = {};
  set_params_fprop(
      params,
      batch_size,
      num_heads_qo,
      num_heads_kv,
      seqlen_qo,
      seqlen_kv,
      head_size_qk,
      head_size_vo,
      seqlen_qo_pad,
      seqlen_kv_pad,
      q,
      k,
      v,
      const_cast<at::Tensor&>(out),
      const_cast<at::Tensor&>(logsumexp),
      scale,
      is_causal);

  params.do_ptr = grad_out.data_ptr();
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();

  params.odo_ptr = tensor_odo.data_ptr();
  params.dqaccum_ptr = tensor_dqaccum.data_ptr();
  params.pb_ptr = tensor_pbuff.data_ptr();

  params.do_batch_stride = grad_out.stride(0);
  params.dq_batch_stride = dq.stride(0);
  params.dk_batch_stride = dk.stride(0);
  params.dv_batch_stride = dv.stride(0);
  params.do_head_stride = grad_out.stride(1);
  params.dq_head_stride = dq.stride(1);
  params.dk_head_stride = dk.stride(1);
  params.dv_head_stride = dv.stride(1);
  params.do_row_stride = grad_out.stride(2);
  params.dq_row_stride = dq.stride(2);
  params.dk_row_stride = dk.stride(2);
  params.dv_row_stride = dv.stride(2);
}
