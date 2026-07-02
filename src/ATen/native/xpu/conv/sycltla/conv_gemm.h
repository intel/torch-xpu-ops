/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * Shared GEMM runner for conv ops. Supports BF16 and FP16.
 * Templated on ElementType (bfloat16_t or half_t).
 */

#pragma once

// sycl-tla headers trigger many warnings that conflict with -Werror.
// Use system_header to suppress all warnings from included headers.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC system_header
#endif

#include <cute/config.hpp>
#include <cute/util/xe_split_barrier.hpp>

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/xe_epilogue.hpp>
#include <cutlass/epilogue/fusion/xe_callbacks.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <sycl/sycl.hpp>

#include <ATen/ATen.h>

using namespace cute;

// ---------------------------------------------------------------------------
// Templated GEMM kernel definition
// ---------------------------------------------------------------------------

template <typename ElementT>
struct ConvGemm {
  using ElementInputA = ElementT;
  using ElementInputB = ElementT;
  using ElementOutput = ElementT;
  using ElementAcc = float;
  using ElementCompute = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using TileShape = Shape<_256, _256, _32>;
  using SubgroupLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  // Select MMA atom based on type
  using MMAAtom = std::conditional_t<
      std::is_same_v<ElementT, bfloat16_t>,
      XE_DPAS_TT<8, float, cute::bfloat16_t>,
      XE_DPAS_TT<8, float, cute::half_t>>;

  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<MMAAtom>,
      Layout<TileShape>,
      SubgroupLayout>::TiledMMA;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      ElementCompute,
      ElementAcc,
      ElementAcc,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape,
      decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape,
      void,
      ElementAcc,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      ElementOutput,
      cutlass::gemm::TagToStrideC_t<LayoutD>,
      FusionCallBacks,
      void,
      void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape,
      ElementInputA,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      ElementInputB,
      cutlass::gemm::TagToStrideB_t<LayoutB>,
      TiledMma,
      void,
      void,
      void,
      cute::identity,
      void,
      void,
      void,
      cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static at::Tensor run(
      sycl::queue& queue,
      const at::Tensor& a,
      const at::Tensor& b) {
    const int M = static_cast<int>(a.size(0));
    const int K = static_cast<int>(a.size(1));
    const int N = static_cast<int>(b.size(1));
    const int L = 1;

    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;

    StrideA stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    StrideB stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    StrideC stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    StrideD stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    at::Tensor out = at::empty({M, N}, a.options());

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, L},
        {static_cast<const ElementInputA*>(a.data_ptr()),
         stride_A,
         static_cast<const ElementInputB*>(b.data_ptr()),
         stride_B},
        {{1.0f, 0.0f},
         nullptr,
         stride_C,
         static_cast<ElementOutput*>(out.data_ptr()),
         stride_D},
        cutlass::KernelHardwareInfo{}};

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    at::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
      workspace = at::empty(
          {static_cast<int64_t>(workspace_size)},
          at::device(at::kXPU).dtype(at::kByte));
      workspace_ptr = workspace.data_ptr();
    }

    auto can_impl = gemm_op.can_implement(arguments);
    TORCH_CHECK(
        can_impl == cutlass::Status::kSuccess,
        "sycltla conv GEMM: invalid size M=",
        M,
        " K=",
        K,
        " N=",
        N);

    auto init_status = gemm_op.initialize(arguments, workspace_ptr);
    TORCH_CHECK(
        init_status == cutlass::Status::kSuccess,
        "sycltla conv GEMM init failed");

    auto run_status = gemm_op.run(&queue);
    TORCH_CHECK(
        run_status == cutlass::Status::kSuccess,
        "sycltla conv GEMM run failed");

    return out;
  }
};

// Dispatch GEMM based on dtype
inline at::Tensor run_conv_gemm(
    sycl::queue& queue,
    const at::Tensor& a,
    const at::Tensor& b) {
  if (a.dtype() == at::kBFloat16) {
    return ConvGemm<bfloat16_t>::run(queue, a, b);
  } else {
    return ConvGemm<half_t>::run(queue, a, b);
  }
}
