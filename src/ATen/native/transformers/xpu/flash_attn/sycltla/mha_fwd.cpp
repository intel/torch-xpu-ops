/*
 * Copyright 2020-2025 Intel Corporation
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

#include "mha_fwd.h"
#include "mha_common.h"

// batch, numhead_qo,numhead_kv,seqlen_qo,seqlen_kv,headsize_qk,headsize_vo
using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>;

namespace cute {

template <class...>
class MhaName;

template <class FMHAPrefillKernel>
struct FA2Runner {
  using StrideQ = typename FMHAPrefillKernel::StrideQ;
  using StrideK = typename FMHAPrefillKernel::StrideK;
  using StrideV = typename FMHAPrefillKernel::StrideV;
  using StrideO = typename FMHAPrefillKernel::StrideO;

  using ElementQ = typename FMHAPrefillKernel::ElementQ;
  using ElementK = typename FMHAPrefillKernel::ElementK;
  using ElementV = typename FMHAPrefillKernel::ElementV;
  using ElementAcc = typename FMHAPrefillKernel::ElementAccumulator;

  using CollectiveEpilogue = typename FMHAPrefillKernel::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;
  using ElementAccumulator = typename CollectiveEpilogue::ElementAccumulator;

  using ProblemShapeType = typename FMHAPrefillKernel::ProblemShape;

  //
  // Methods
  //

  // Note that the GemmUniversalAdapter currently doesn't support flash
  // attention, which is why this secondary `run` function is required to launch
  // the kernel.
  void run(sycl::queue& queue, typename FMHAPrefillKernel::Params params) {
    dim3 const block = FMHAPrefillKernel::get_block_shape();
    dim3 const grid = FMHAPrefillKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAPrefillKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

// Launch parameters depend on whether SYCL compiler supports work-group scratch
// memory extension
#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
    using namespace compat::experimental;
    auto event = launch<
        cutlass::device_kernel<FMHAPrefillKernel>,
        MhaName<FMHAPrefillKernel>>(
        launch_policy{
            sycl_grid,
            sycl_block,
            local_mem_size{static_cast<std::size_t>(smem_size)},
            kernel_properties{sycl_exp::sub_group_size<
                FMHAPrefillKernel::DispatchPolicy::SubgroupSize>}},
        queue,
        params);
#else
    compat::experimental::launch_properties launch_props{
        sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        sycl::ext::oneapi::experimental::sub_group_size<
            FMHAPrefillKernel::DispatchPolicy::SubgroupSize>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = compat::experimental::launch<
        cutlass::device_kernel<FMHAPrefillKernel>,
        MhaName<FMHAPrefillKernel>>(policy, queue, params);
#endif
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

    ProblemShapeType problem_size{
        batch,
        num_heads_qo,
        num_heads_kv,
        seq_len_qo,
        seq_len_kv,
        head_size_qk,
        head_size_vo};

    const ElementQ* inputQ = static_cast<const ElementQ*>(params.q_ptr);
    const ElementK* inputK = static_cast<const ElementK*>(params.k_ptr);
    const ElementV* inputV = static_cast<const ElementV*>(params.v_ptr);
    ElementOutput* output = static_cast<ElementOutput*>(params.o_ptr);
    float* logsumexp = static_cast<float*>(params.lse_ptr);
    float softmax_scale = params.scale;

    typename FMHAPrefillKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        {inputQ,
         params.q_batch_stride,
         params.q_head_stride,
         params.q_row_stride,
         inputK,
         params.k_batch_stride,
         params.k_head_stride,
         params.k_row_stride,
         inputV,
         params.v_batch_stride,
         params.v_head_stride,
         params.v_row_stride},
        {softmax_scale},
        {output,
         params.o_batch_stride,
         params.o_head_stride,
         params.o_row_stride,
         logsumexp},
        hw_info,
        softmax_scale};

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
    typename SubgroupLayout,
    int PipelineStages>
void run_mha_fwd_(sycl::queue& queue, FLASH_FWD_params& params) {
  cutlass::KernelHardwareInfo hw_info;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;

  using ElementInputQ = T;
  using ElementInputKV = T;
  using ElementOutput = T;
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;

  using MMAOperation = std::conditional_t<
      std::is_same_v<T, bfloat16_t>,
      XE_8x16x16_F32BF16BF16F32_TT,
      XE_8x16x16_F32F16F16F32_TT>;
  using GmemTiledCopyQ = XE_2D_U16x8x32_LD_N;
  using GmemTiledCopyK =
      XE_2D_U16x16x16_LD_T; // _T designates a transposed block load operation
  using GmemTiledCopyV = XE_2D_U16x16x32_LD_V;
  using GmemTiledCopyStore = XE_2D_U16x8x16_ST_N; // Change to output BF16

  using GEMMDispatchPolicy =
      cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;
  using CollectiveEpilogue =
      cutlass::flash_attention::collective::FlashPrefillEpilogue<
          EpilogueDispatchPolicy,
          MMAOperation,
          TileShapeOutPut,
          SubgroupLayout,
          ElementComputeEpilogue,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutO>,
          ElementOutput,
          GmemTiledCopyStore>;
  using CollectiveSoftmaxEpilogue =
      cutlass::flash_attention::collective::FlashPrefillSoftmaxEpilogue<
          IS_CAUSAL,
          EpilogueDispatchPolicy,
          ElementAccumulator>;

  using namespace cutlass::fmha::collective;

  using ProblemShapeType = ProblemShapeRegular;

  // Mainloop
  using CollectiveMainloop =
      cutlass::flash_attention::collective::FlashPrefillMma<
          GEMMDispatchPolicy,
          ProblemShapeType,
          ElementInputQ,
          cutlass::gemm::TagToStrideA_t<LayoutQ>,
          ElementInputKV,
          cutlass::gemm::TagToStrideB_t<LayoutK>,
          ElementInputKV,
          cutlass::gemm::TagToStrideB_t<LayoutV>,
          MMAOperation,
          TileShapeQK,
          TileShapePV,
          SubgroupLayout,
          GmemTiledCopyQ, // Q
          GmemTiledCopyK, // K
          GmemTiledCopyV, // V,
          IS_CAUSAL>;
  using FMHAPrefillKernel = cutlass::flash_attention::kernel::FMHAPrefill<
      ProblemShapeType,
      CollectiveMainloop,
      CollectiveSoftmaxEpilogue,
      CollectiveEpilogue,
      cutlass::flash_attention::IndividualScheduler>;

  FA2Runner<FMHAPrefillKernel> runner;
  runner.run(queue, params, hw_info);
}

template <typename T, bool IS_CAUSAL>
void run_mha_fwd_(sycl::queue& queue, FLASH_FWD_params& params) {
  const int headdim = params.head_size_vo;

#define run_mha_fwd_specialized( \
    TileShapeQK_,                \
    TileShapePV_,                \
    TileShapeOutPut_,            \
    SubgroupLayout_,             \
    PipelineStages_)             \
  run_mha_fwd_<                  \
      T,                         \
      IS_CAUSAL,                 \
      TileShapeQK_,              \
      TileShapePV_,              \
      TileShapeOutPut_,          \
      SubgroupLayout_,           \
      PipelineStages_>(queue, params);

  if (headdim == 64) {
    constexpr int PipelineStages = 2;
    using TileShapeQK = Shape<_128, _64, _64>;
    using TileShapePV = Shape<_128, _32, _64>;
    using TileShapeOutPut = Shape<_128, _64, _64>;
    using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayout,
        PipelineStages);
  } else if (headdim == 96) {
    constexpr int PipelineStages = 2;
    using TileShapeQK = Shape<_128, _64, _32>;
    using TileShapePV = Shape<_128, _32, _64>;
    using TileShapeOutPut = Shape<_128, _96, _64>;
    using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayout,
        PipelineStages);
  } else if (headdim == 128) {
    auto device_architecture =
        queue.get_device()
            .get_info<
                sycl::ext::oneapi::experimental::info::device::architecture>();
    if (device_architecture ==
            sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc ||
        device_architecture ==
            sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc_vg) {
      constexpr int PipelineStages = 1;
      using TileShapeQK = Shape<_64, _32, _64>;
      using TileShapePV = Shape<_64, _32, _32>;
      using TileShapeOutPut = Shape<_64, _128, _32>;
      using SubgroupLayout = Layout<Shape<_4, _1, _1>, Stride<_1, _1, _1>>;
      run_mha_fwd_specialized(
          TileShapeQK,
          TileShapePV,
          TileShapeOutPut,
          SubgroupLayout,
          PipelineStages);
    } else {
      constexpr int PipelineStages = 2;
      using TileShapeQK = Shape<_256, _32, _64>;
      using TileShapePV = Shape<_256, _32, _32>;
      using TileShapeOutPut = Shape<_256, _128, _32>;
      using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
      run_mha_fwd_specialized(
          TileShapeQK,
          TileShapePV,
          TileShapeOutPut,
          SubgroupLayout,
          PipelineStages);
    }
  } else if (headdim == 192) {
    constexpr int PipelineStages = 2;
    using TileShapeQK = Shape<_256, _64, _64>;
    using TileShapePV = Shape<_256, _32, _64>;
    using TileShapeOutPut = Shape<_256, _192, _64>;
    using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
    run_mha_fwd_specialized(
        TileShapeQK,
        TileShapePV,
        TileShapeOutPut,
        SubgroupLayout,
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

  auto opts = query.options();
  at::Tensor out = at::empty_like(query);

  if (seqlen_qo > seqlen_kv && is_causal) {
    // When seqlen_qo is greater than seqlen_kv and is_causal(lower_right causal
    // mask) is true, some output positions will skip computation for better
    // performance.
    out.zero_();
  }

  at::Tensor logsumexp =
      at::empty({batch_size, numhead_qo, seqlen_qo}, opts.dtype(at::kFloat));

  auto sycl_queue = at::xpu::getCurrentXPUStream().queue();
  auto device_architecture =
      sycl_queue.get_device()
          .get_info<
              sycl::ext::oneapi::experimental::info::device::architecture>();
  constexpr auto supported_architectures =
      std::array<sycl::ext::oneapi::experimental::architecture, 3>{
          sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_pvc_vg,
          sycl::ext::oneapi::experimental::architecture::intel_gpu_bmg_g21};
  if (std::find(
          supported_architectures.begin(),
          supported_architectures.end(),
          device_architecture) == supported_architectures.end()) {
    TORCH_CHECK(
        false,
        "XPU device architecture does not support flash attention. Supported architectures are: intel_gpu_pvc, intel_gpu_pvc_vg, intel_gpu_bmg_g21.");
  }

  FLASH_FWD_params params;
  set_params_fprop(
      params,
      batch_size,
      numhead_qo,
      numhead_kv,
      seqlen_qo,
      seqlen_kv,
      headsize_qk,
      headsize_vo,
      0, // unused seqlen_qo_pad
      0, // unused seqlen_kv_pad
      query,
      key,
      value,
      out,
      logsumexp,
      scale,
      is_causal);

  cute::run_mha_fwd(sycl_queue, params);

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
      /* max_seqlen_batch_q */ c10::SymInt(0),
      /* max_seqlen_batch_k */ c10::SymInt(0),
      /* philox_seed */ at::empty({}, at::dtype(at::kLong)),
      /* philox_offset */ at::empty({}, at::dtype(at::kLong))};
}

} // namespace sycltla
