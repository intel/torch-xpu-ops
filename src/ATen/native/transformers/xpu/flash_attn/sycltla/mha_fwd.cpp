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

// batch, numhead_qo,numhead_kv,seqlen_qo,seqlen_kv,headsize_qk,headsize_vo
using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>;

namespace cute {

template <class...>
class MhaName;

template <class FMHAPrefillKernel, bool isVarLen>
struct FA2Runner {
  using ElementQ = typename FMHAPrefillKernel::ElementQ;
  using ElementK = typename FMHAPrefillKernel::ElementK;
  using ElementV = typename FMHAPrefillKernel::ElementV;
  using ElementO = typename FMHAPrefillKernel::ElementO;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;

  //
  // Methods
  //

  // Note that the GemmUniversalAdapter currently doesn't support flash
  // attention, which is why this secondary `run` function is required to launch
  // the kernel.
  void run(sycl::queue& queue, typename FMHAPrefillKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAPrefillKernel::get_block_shape();
    dim3 const grid = FMHAPrefillKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAPrefillKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group
    // scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    compat::experimental::launch<
        cutlass::device_kernel<FMHAPrefillKernel>,
        MhaName<FMHAPrefillKernel>>(policy, queue, params);
  }

  void run(
      sycl::queue& queue,
      FLASH_FWD_params& params,
      const cutlass::KernelHardwareInfo& hw_info) {
    int batch = params.batch_size;
    int num_heads_qo = params.num_heads_qo;
    int num_heads_kv = params.num_heads_kv;
    int seq_len_qo = params.seqlen_qo;
    int seq_len_kv = params.seqlen_kv;
    int head_size_qk = params.head_size_qk;
    int head_size_vo = params.head_size_vo;

    ProblemShapeType shape;
    shape.batch = batch;
    shape.num_heads_q = num_heads_qo;
    shape.num_heads_kv = num_heads_kv;
    shape.seq_len_qo = seq_len_qo;
    shape.seq_len_kv = seq_len_kv;
    shape.head_size_qk = head_size_qk;
    shape.head_size_vo = head_size_vo;

    const ElementQ* q_ptr = static_cast<const ElementQ*>(params.q_ptr);
    const ElementK* k_ptr = static_cast<const ElementK*>(params.k_ptr);
    const ElementV* v_ptr = static_cast<const ElementV*>(params.v_ptr);
    ElementO* o_ptr = static_cast<ElementO*>(params.o_ptr);
    float* lse_ptr = static_cast<float*>(params.lse_ptr);
    float softmax_scale = params.scale;

    typename FMHAPrefillKernel::Arguments arguments{
        {
            shape,
            q_ptr,
            params.q_batch_stride,
            params.q_head_stride,
            params.q_row_stride,
            k_ptr,
            params.k_batch_stride,
            params.k_head_stride,
            params.k_row_stride,
            v_ptr,
            params.v_batch_stride,
            params.v_head_stride,
            params.v_row_stride,
            o_ptr,
            params.o_batch_stride,
            params.o_head_stride,
            params.o_row_stride,
            lse_ptr,
        },
        {softmax_scale},
        {},
        hw_info};

    // Define device-global scratch memory
    size_t workspace_size = FMHAPrefillKernel::get_workspace_size(arguments);
    at::Tensor workspace_tensor = at::empty(
        {static_cast<int64_t>(workspace_size)},
        at::device(at::kXPU).dtype(at::kByte));

    if (!FMHAPrefillKernel::can_implement(arguments)) {
      TORCH_CHECK(
          false,
          "Invalid Problem Size",
          batch,
          "x",
          num_heads_qo,
          "x",
          seq_len_qo,
          "x",
          seq_len_kv,
          "x",
          head_size_qk,
          "x",
          head_size_vo);
      return;
    }

    // Initialize the workspace
    CUTLASS_CHECK(FMHAPrefillKernel::initialize_workspace(
        arguments, workspace_tensor.data_ptr()));

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto kernel_params = FMHAPrefillKernel::to_underlying_arguments(
        arguments, workspace_tensor.data_ptr());

    // Launch a SYCL kernel using scratch/shared memory
    run(queue, kernel_params);
  }
};

template <
    typename T,
    bool IS_CAUSAL,
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutPut,
    typename SubgroupLayoutQK,
    int PipelineStages,
    bool isVarLen = false>
void run_mha_fwd_(sycl::queue& queue, FLASH_FWD_params& params) {
  using ElementQ = T;
  using ElementK = T;
  using ElementV = T;
  using ElementO = T;
  using StrideQ = Stride<int64_t, _1, int64_t, int64_t>;
  using StrideK = Stride<int64_t, _1, int64_t, int64_t>;
  using StrideV = Stride<_1, int64_t, int64_t, int64_t>;
  using StrideO = Stride<int64_t, _1, int64_t, int64_t>;
  auto make_dummy_tensor = [&](auto val, auto stride) {
    return make_tensor(
        make_gmem_ptr(static_cast<decltype(val)*>(nullptr)),
        make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };
  auto make_const_dummy_tensor = [&](auto val, auto stride) {
    return make_tensor(
        make_gmem_ptr(static_cast<const decltype(val)*>(nullptr)),
        make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
  };
  using TensorQ = decltype(make_const_dummy_tensor(ElementQ{}, StrideQ{}));
  using TensorK = decltype(make_const_dummy_tensor(ElementK{}, StrideK{}));
  using TensorV = decltype(make_const_dummy_tensor(ElementV{}, StrideV{}));
  using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));

  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  static_assert(SGTileQ <= 16, "Subgroup tile in Q dimension must be <= 16");
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, T>;
  using SubgroupLayoutPV =
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));
  using TiledMMAQK = typename TiledMMAHelper<
      MMA_Atom<MMAOperation>,
      Layout<TileShapeQK>,
      SubgroupLayoutQK>::TiledMMA;
  using TiledMMAPV = typename TiledMMAHelper<
      MMA_Atom<MMAOperation>,
      Layout<TileShapePV>,
      SubgroupLayoutPV>::TiledMMA;
  static_assert(
      get<0>(TileShapeOutPut{}) == get<0>(TileShapePV{}),
      "Output tile and P*V tile have different sizes in Q dimension");
  constexpr int VTiles = get<1>(TileShapeOutPut{}) / get<1>(TileShapePV{});

  cutlass::KernelHardwareInfo hw_info;

  // Mainloop
  using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
  using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
      MainloopDispatchPolicy,
      IS_CAUSAL,
      TiledMMAQK,
      TiledMMAPV,
      VTiles,
      TensorQ,
      TensorK,
      TensorV>;

  // Epilogue
  using CollectiveEpilogue = cutlass::fmha::collective::
      FMHAFwdEpilogue<CollectiveMainloop, TileShapeOutPut, TensorO>;

  using Scheduler = cutlass::fmha::kernel::XeFMHAIndividualTileScheduler;
  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;
  using FMHAPrefillKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
      ProblemShapeType,
      CollectiveMainloop,
      CollectiveEpilogue,
      Scheduler>;

  FA2Runner<FMHAPrefillKernel, isVarLen> runner;
  runner.run(queue, params, hw_info);
}

template <typename T, bool IS_CAUSAL>
void run_mha_fwd_(sycl::queue& queue, FLASH_FWD_params& params) {
  const int headdim = params.head_size_vo;

#define run_mha_fwd_specialized( \
    TileShapeQK_,                \
    TileShapePV_,                \
    TileShapeOutPut_,            \
    SubgroupLayoutQK_,           \
    PipelineStages_)             \
  run_mha_fwd_<                  \
      T,                         \
      IS_CAUSAL,                 \
      TileShapeQK_,              \
      TileShapePV_,              \
      TileShapeOutPut_,          \
      SubgroupLayoutQK_,         \
      PipelineStages_>(queue, params);

  constexpr int PipelineStages = 2;
  if (headdim == 64) {
    int64_t batch_size = params.batch_size;
    int64_t num_heads_qo = params.num_heads_qo;
    int64_t seqlen_qo = params.seqlen_qo;
    if (batch_size * num_heads_qo * seqlen_qo <= 8192) {
      using TileShapeQK = Shape<_64, _64, _32>;
      using TileShapePV = Shape<_64, _32, _64>;
      using TileShapeOutPut = Shape<_64, _64>;
      using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
      run_mha_fwd_specialized(
          TileShapeQK,
          TileShapePV,
          TileShapeOutPut,
          SubgroupLayoutQK,
          PipelineStages);
    } else {
      using TileShapeQK = Shape<_128, _64, _32>;
      using TileShapePV = Shape<_128, _32, _64>;
      using TileShapeOutPut = Shape<_128, _64>;
      using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
      run_mha_fwd_specialized(
          TileShapeQK,
          TileShapePV,
          TileShapeOutPut,
          SubgroupLayoutQK,
          PipelineStages);
    }
  } else if (headdim == 96) {
    using TileShapeQK = Shape<_128, _64, _32>;
    using TileShapePV = Shape<_128, _32, _64>;
    using TileShapeOutPut = Shape<_128, _96>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayoutQK,
        PipelineStages);
  } else if (headdim == 128) {
    using TileShapeQK = Shape<_128, _32, _32>;
    using TileShapePV = Shape<_128, _32, _32>;
    using TileShapeOutPut = Shape<_128, _128>;
    using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayoutQK,
        PipelineStages);
  } else if (headdim == 192) {
    using TileShapeQK = Shape<_256, _64, _32>;
    using TileShapePV = Shape<_256, _32, _64>;
    using TileShapeOutPut = Shape<_256, _192>;
    using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayoutQK,
        PipelineStages);
  } else {
    TORCH_CHECK(
        false, "FlashAttentionForwardXPU only support headdim 64,96,128,192");
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
      p_dropout == 0.0, "mha_fwd on xpu does not support p_dropout > 0.0 yet");
  TORCH_CHECK(
      alibi_slopes_.has_value() == false,
      "mha_fwd on xpu does not support alibi_slopes yet");
  TORCH_CHECK(
      window_size_left == -1 && window_size_right == -1,
      "mha_fwd on xpu does not support window_size yet");
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
        at::empty({0, seqlen_kv, numhead_kv, headsize_qk}, opts);
    at::Tensor v_padded =
        at::empty({0, seqlen_kv, numhead_kv, headsize_vo}, opts);
    at::Tensor softmax_lse =
        at::empty({0, numhead_qo, seqlen_qo}, opts.dtype(at::kFloat));
    at::Tensor philox_seed = at::empty({}, opts.dtype(at::kLong));
    at::Tensor philox_offset = at::empty({}, opts.dtype(at::kLong));
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
      headsize_qk <= 192,
      "mha_fwd on xpu: only support head dimension at most 192");

  const int headsize_padded = headsize_qk;
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
    CHECK_SHAPE(out, batch_size, seqlen_qo, numhead_qo, headsize_vo);
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