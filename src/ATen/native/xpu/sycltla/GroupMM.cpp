/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "GroupMM.h"
#include "GroupMMCommon.h"

#include <ATen/native/GroupedMMUtils.h>
#include <ATen/xpu/XPUContext.h>

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/xe_array_epilogue.hpp>
#include <cutlass/epilogue/fusion/xe_callbacks.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>

namespace at::xpu::sycltla {

namespace {

#define CUTLASS_CHECK(status)                                                  \
  do {                                                                         \
    cutlass::Status error = status;                                            \
    TORCH_CHECK(                                                                \
        error == cutlass::Status::kSuccess,                                    \
        "bf16bf16_grouped_mm CUTLASS error: ",                                \
        cutlassGetStatusString(error));                                        \
  } while (false)

using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using ElementInput = cute::bfloat16_t;
using ElementOutput = float;
using ElementAccumulator = float;
using ElementComputeEpilogue = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using GmemTiledCopyA = void;
using GmemTiledCopyB = void;
using TileShape = cute::Shape<cute::_256, cute::_256, cute::_32>;

constexpr int kPipelineStages = 2;
using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<kPipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput,
    ElementComputeEpilogue,
    ElementAccumulator,
    ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
using ElementA = ElementInput;
using ElementB = ElementInput;
using TiledMma = typename cute::TiledMMAHelper<
    cute::MMA_Atom<cute::XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
    cute::Layout<TileShape>,
    cute::Layout<cute::Shape<cute::_8, cute::_4, cute::_1>,
                 cute::Stride<cute::_4, cute::_1, cute::_0>>>::TiledMMA;
using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy,
    EpilogueOp,
    TileShape,
    decltype(cute::tile_shape(TiledMma()))>;
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC*>,
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD*>,
    FusionCallbacks,
    cute::XE_2D_U32x8x16_LD_N,
    void,
    void,
    cute::XE_2D_U32x8x16_ST_N,
    void,
    void>;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementA,
    cutlass::gemm::TagToStrideA_t<LayoutA*>,
    ElementB,
    cutlass::gemm::TagToStrideB_t<LayoutB*>,
    TiledMma,
    GmemTiledCopyA,
    void,
    void,
    cute::identity,
    GmemTiledCopyB,
    void,
    void,
    cute::identity>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::GroupScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

struct GroupedMMArguments {
  const at::Tensor* mat_a;
  const at::Tensor* mat_b;
  const std::optional<at::Tensor>* offs;
  const std::optional<at::Tensor>* bias;
  at::Tensor* out;
};

struct GroupDesc {
  at::Tensor a;
  at::Tensor b;
  at::Tensor dst;
};

struct GroupedMMRunner {
  static bool can_implement(const GroupedMMArguments& args) {
    if (args.bias->has_value()) {
      return false;
    }
    if (args.mat_a->dtype() != at::kBFloat16 || args.mat_b->dtype() != at::kBFloat16 ||
        args.out->dtype() != at::kBFloat16) {
      return false;
    }
    if (args.mat_a->dim() < 2 || args.mat_a->dim() > 3 || args.mat_b->dim() < 2 ||
        args.mat_b->dim() > 3) {
      return false;
    }
    if ((args.mat_a->dim() == 2 || args.mat_b->dim() == 2) && !args.offs->has_value()) {
      return false;
    }
    const auto group_count =
        args.mat_a->dim() == 3 ? args.mat_a->size(0) : args.offs->value().size(0);
    return group_count < 1024;
  }

  static std::vector<GroupDesc> build_groups(const GroupedMMArguments& args) {
    std::vector<GroupDesc> groups;
    const auto& mat_a = *args.mat_a;
    const auto& mat_b = *args.mat_b;
    auto& out = *args.out;

    const bool a_is_2d = mat_a.dim() == 2;
    const bool b_is_2d = mat_b.dim() == 2;

    if (a_is_2d && !b_is_2d) {
      int64_t group_start_idx = 0;
      auto offs_cpu = args.offs->value().cpu();
      groups.reserve(static_cast<size_t>(offs_cpu.size(0)));
      for (int64_t group_idx = 0; group_idx < offs_cpu.size(0); ++group_idx) {
        int64_t group_end_idx = offs_cpu[group_idx].item<int>();
        groups.push_back({
            mat_a.slice(0, group_start_idx, group_end_idx),
            mat_b[group_idx],
            out.slice(0, group_start_idx, group_end_idx)});
        group_start_idx = group_end_idx;
      }
      return groups;
    }

    if (!a_is_2d && b_is_2d) {
      int64_t group_start_idx = 0;
      auto offs_cpu = args.offs->value().cpu();
      groups.reserve(static_cast<size_t>(offs_cpu.size(0)));
      for (int64_t group_idx = 0; group_idx < offs_cpu.size(0); ++group_idx) {
        int64_t group_end_idx = offs_cpu[group_idx].item<int>();
        groups.push_back({
            mat_a[group_idx],
            mat_b.slice(1, group_start_idx, group_end_idx),
            out.slice(1, group_start_idx, group_end_idx)});
        group_start_idx = group_end_idx;
      }
      return groups;
    }

    if (a_is_2d && b_is_2d) {
      int64_t group_start_idx = 0;
      auto offs_cpu = args.offs->value().cpu();
      groups.reserve(static_cast<size_t>(offs_cpu.size(0)));
      for (int64_t group_idx = 0; group_idx < offs_cpu.size(0); ++group_idx) {
        int64_t group_end_idx = offs_cpu[group_idx].item<int>();
        groups.push_back({
            mat_a.slice(1, group_start_idx, group_end_idx),
            mat_b.slice(0, group_start_idx, group_end_idx),
            out[group_idx]});
        group_start_idx = group_end_idx;
      }
      return groups;
    }

    groups.reserve(static_cast<size_t>(mat_a.size(0)));
    for (int64_t i = 0; i < mat_a.size(0); ++i) {
      groups.push_back({mat_a[i], mat_b[i], out[i]});
    }
    return groups;
  }

  static void run_fallback(const GroupedMMArguments& args) {
    at::native::_grouped_mm_fallback(
        *args.mat_a,
        *args.mat_b,
        *args.offs,
        *args.bias,
        args.out->scalar_type(),
        *args.out);
  }

  static bool run_grouped_kernel_impl(const GroupedMMArguments& args) {
    auto groups = build_groups(args);
    if (groups.empty()) {
      return true;
    }

    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    using PreparedData = at::xpu::sycltla::detail::GroupedGemmData<
        ProblemShape,
        ElementA,
        ElementB,
        ElementOutput,
        StrideA,
        StrideB,
        StrideC,
        StrideD>;
    PreparedData prepared;
    if (!at::xpu::sycltla::detail::prepare_grouped_gemm_data(groups, prepared)) {
      return false;
    }
    const auto group_count = groups.size();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    Gemm gemm_op;
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 1.0f;
    fusion_args.beta = 0.0f;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<
            ProblemShape>::RasterOrderOptions;

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
      {static_cast<int>(group_count), prepared.problem_sizes.get(), prepared.problem_sizes_host.data()},
      {prepared.ptr_A.get(), prepared.stride_A.get(), prepared.ptr_B.get(), prepared.stride_B.get()},
      {fusion_args,
       prepared.ptr_C.get(),
       prepared.stride_C.get(),
       prepared.ptr_D.get(),
       prepared.stride_D.get()},
        hw_info,
        {1, RasterOrderOptions::AlongN}};

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op.run());

    for (size_t i = 0; i < group_count; ++i) {
      groups[i].dst.copy_(prepared.packed_d[i].to(groups[i].dst.scalar_type()));
    }
    return true;
  }

  static bool run_grouped_kernel(const GroupedMMArguments& args) {
    return run_grouped_kernel_impl(args);
  }

  static void run(const GroupedMMArguments& args) {
    if (!run_grouped_kernel(args)) {
      run_fallback(args);
    }
  }
};

} // namespace

} // namespace at::xpu::sycltla

namespace at::xpu::detail {

void bf16bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias,
    at::Tensor& out) {
  at::xpu::sycltla::GroupedMMArguments args{&mat_a, &mat_b, &offs, &bias, &out};
  TORCH_CHECK(
      at::xpu::sycltla::GroupedMMRunner::can_implement(args),
      "bf16bf16_grouped_mm: unsupported input combination");
  at::xpu::sycltla::GroupedMMRunner::run(args);
}

} // namespace at::xpu::detail
