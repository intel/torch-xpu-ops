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
 * Fused Dual GEMM with SiLU-Mul epilogue for XPU (Intel GPU / PVC).
 *
 * Computes: out = silu(x @ w1.T) * (x @ w3.T)
 *
 * Uses the sycl-tla (CUTLASS SYCL port, v0.6+) DualGemm kernel which loads
 * the shared A matrix (x) from HBM only once, saving one full matrix read
 * compared to two separate GEMM calls.
 *
 * Tile configuration: 128×128×64 with WarpLayout 8×4×1 (32 subgroups).
 * This is the only tile shape compatible with sycl-tla's DualGemm MMA/copy
 * atom constraints (make_fragment_layout requires specific alignment between
 * GmemTiledCopy block shapes and MMA atom fragment sizes).
 *
 * NOTE: For LLaMA inference shapes (small M, large N), the fused kernel is
 * slower than eager (oneDNN) due to M-dimension padding overhead.  The FX
 * fusion pass (xpu_dual_gemm.py) includes shape guards to skip fusion in
 * those cases.  The fused kernel is beneficial when M is large enough to
 * fill the 128-row tiles efficiently (roughly M >= 512 for square-ish shapes).
 */

// dual_gemm_common.h is in the parent directory (dual_gemm/).
// The build system adds that directory to the include path.
#include <sycltla/dual_gemm_common.h>

// gemm_universal_adapter.h must come first: it includes gemm_universal.hpp
// which in turn includes tile_scheduler.hpp (defines PersistentScheduler and
// detail::TileSchedulerSelector).  xe_dual_gemm.hpp depends on these but does
// not include tile_scheduler.hpp itself.
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/packed_stride.hpp>
#include <sycl/sycl.hpp>

#include <dual_gemm/collective/xe_dual_gemm_epilogue.hpp>
#include <dual_gemm/collective/xe_dual_gemm_epilogue_elementwise_activation.hpp>
#include <dual_gemm/kernel/xe_dual_gemm.hpp>
#include <dual_gemm/thread/xe_binary_elem_wise_op.hpp>

using namespace cute;

// ---------------------------------------------------------------------------
// Type aliases – BF16 input / BF16 output, float accumulator
// ---------------------------------------------------------------------------

using ElementInputAB = bfloat16_t;
using ElementOutput = bfloat16_t;
using ElementAcc = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor; // weights are [N, K], row-major
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using GmemTiledCopyA = XE_2D_U16x16x32_LD_N;
using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;

using TileShape = Shape<_128, _128, _64>;

using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
    Layout<TileShape>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

constexpr int PipelineStages = 2;
using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

// Epilogue 0: Identity (no activation) – we'll apply activation in the fused
// ElemAct epilogue that writes the final output.  We still need to write the
// intermediate D0/D1 so the ElemAct epilogue can read them.
//
// Actually with DualGemmElemActEpilogue the intermediate outputs are kept in
// registers / SLM – we don't need to write them to global memory.  Setting
// WriteEpilogueOutput = false keeps them in-registers.
constexpr bool WriteEpilogueOutput0 = false;
constexpr bool WriteEpilogueOutput1 = false;

using EpilogueOp0 = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementCompute,
    ElementAcc,
    ElementAcc,
    cutlass::FloatRoundStyle::round_to_nearest>;

using EpilogueOp1 = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::Identity,
    ElementOutput,
    ElementCompute,
    ElementAcc,
    ElementAcc,
    cutlass::FloatRoundStyle::round_to_nearest>;

using FusionCallBacks0 = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy,
    EpilogueOp0,
    TileShape,
    decltype(tile_shape(TiledMma()))>;

using FusionCallBacks1 = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy,
    EpilogueOp1,
    TileShape,
    decltype(tile_shape(TiledMma()))>;

// ElementC must equal the mainloop ElementAccumulator (float) due to the
// static_assert in xe_dual_gemm.hpp.  The C bias is never actually loaded
// (ptr_C = nullptr, beta = 0), but the type must still be consistent.
// XE_2D_U32x8x16_LD_N handles 32-bit (float) loads; XE_2D_U16x8x16_ST_N
// handles 16-bit (bfloat16) stores for D.
// WriteEpilogueOutput = false means D0/D1 are never written to global memory;
// they stay in registers for the ElemAct epilogue.
using CollectiveEpilogue0 = cutlass::epilogue::collective::DualGemmEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    ElementAcc, // ElementC = float (= ElementAccumulator, required by mainloop)
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput, // ElementD = bfloat16_t
    cutlass::gemm::TagToStrideC_t<LayoutD>,
    FusionCallBacks0,
    XE_2D_U32x8x16_LD_N, // 32-bit load for float C
    XE_2D_U16x8x16_ST_N, // 16-bit store for bf16 D (unused since
                         // WriteOutput=false)
    WriteEpilogueOutput0>;

using CollectiveEpilogue1 = cutlass::epilogue::collective::DualGemmEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    ElementAcc, // ElementC = float
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput, // ElementD = bfloat16_t
    cutlass::gemm::TagToStrideC_t<LayoutD>,
    FusionCallBacks1,
    XE_2D_U32x8x16_LD_N,
    XE_2D_U16x8x16_ST_N,
    WriteEpilogueOutput1>;

// Fused element-wise epilogue: silu(D0) * D1 → D2
using EpilogueOutputOp2 = cutlass::epilogue::thread::FusedElementWiseOpDualGemm<
    ElementOutput,
    cutlass::epilogue::thread::SiLu,
    cutlass::epilogue::thread::Identity,
    cutlass::multiplies,
    ElementAcc,
    ElementAcc>;

using CollectiveEpilogueActivation =
    cutlass::epilogue::collective::DualGemmElemActEpilogue<
        EpilogueDispatchPolicy,
        TileShape,
        void,
        cutlass::gemm::TagToStrideC_t<LayoutC>,
        ElementOutput,
        cutlass::gemm::TagToStrideC_t<LayoutD>,
        void,
        XE_2D_U16x8x16_ST_N,
        EpilogueOutputOp2>;

using CollectiveMainloop = cutlass::gemm::collective::DualGemmMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementInputAB,
    cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementInputAB,
    cutlass::gemm::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA,
    GmemTiledCopyB>;

using GemmKernel = cutlass::gemm::kernel::DualGemm<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue0,
    CollectiveEpilogue1,
    CollectiveEpilogueActivation>;

// ---------------------------------------------------------------------------
// Kernel launcher – mirrors the pattern used in mha_fwd.cpp
// ---------------------------------------------------------------------------

// Anonymous name tag for kernel caching (analogous to MhaName in mha_fwd.cpp)
namespace cute {
template <class...>
class DualGemmName;
} // namespace cute

static cutlass::Status launch_dual_gemm_kernel(
    sycl::queue& queue,
    typename GemmKernel::Params params) {
  dim3 const block = GemmKernel::get_block_shape();
  dim3 const grid = GemmKernel::get_grid_shape(params);
  int smem_size = GemmKernel::SharedStorageSize;

  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
  using namespace compat::experimental;
  auto event = launch<
      cutlass::device_kernel<GemmKernel>,
      cute::DualGemmName<GemmKernel>>(
      launch_policy{
          sycl_grid,
          sycl_block,
          local_mem_size{static_cast<std::size_t>(smem_size)},
          kernel_properties{sycl_exp::sub_group_size<
              GemmKernel::DispatchPolicy::SubgroupSize>}},
      queue,
      params);
#else
  compat::experimental::launch_properties launch_props{
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<
          GemmKernel::DispatchPolicy::SubgroupSize>};
  compat::experimental::launch_policy policy{
      sycl_grid, sycl_block, launch_props, kernel_props};
  auto event = compat::experimental::launch<
      cutlass::device_kernel<GemmKernel>,
      cute::DualGemmName<GemmKernel>>(policy, queue, params);
#endif
  (void)event;
  return cutlass::Status::kSuccess;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

namespace sycltla {

at::Tensor dual_gemm_silu_mul_impl(
    sycl::queue& queue,
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w3) {
  // Dimensions: x [M, K], w1 [K, N], w3 [K, N]  (already transposed by caller)
  const int M = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w1.size(1)); // w1 is [K, N]
  const int L = 1; // no batch

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideD = typename GemmKernel::StrideD;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

  // No C matrix – set alpha=1, beta=0
  using EpilogueArguments0 = typename GemmKernel::EpilogueArguments0;
  using EpilogueArguments1 = typename GemmKernel::EpilogueArguments1;

  // C pointers set to nullptr because beta=0 and we skip the C load.
  // We also skip writing D0/D1 (WriteEpilogueOutput = false) so nullptr is
  // safe.
  EpilogueArguments0 epilogue0{
      {1.f, 0.f}, nullptr, stride_D, nullptr, stride_D};
  EpilogueArguments1 epilogue1{
      {1.f, 0.f}, nullptr, stride_D, nullptr, stride_D};

  // Output tensor for silu(x@w1.T) * (x@w3.T)
  at::Tensor out = at::empty({M, N}, x.options());

  typename GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, L},
      {static_cast<const ElementInputAB*>(x.data_ptr()),
       stride_A,
       static_cast<const ElementInputAB*>(w1.data_ptr()),
       stride_B,
       static_cast<const ElementInputAB*>(w3.data_ptr()),
       stride_B},
      epilogue0,
      epilogue1,
      {static_cast<ElementOutput*>(out.data_ptr()), stride_D},
      cutlass::KernelHardwareInfo{}};

  // Workspace
  size_t workspace_size = GemmKernel::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty(
        {static_cast<int64_t>(workspace_size)},
        at::device(at::kXPU).dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  TORCH_CHECK(
      GemmKernel::can_implement(arguments),
      "dual_gemm_silu_mul: invalid problem size M=",
      M,
      " N=",
      N,
      " K=",
      K);

  CUTLASS_CHECK(GemmKernel::initialize_workspace(arguments, workspace_ptr));

  auto params = GemmKernel::to_underlying_arguments(arguments, workspace_ptr);
  CUTLASS_CHECK(launch_dual_gemm_kernel(queue, params));

  return out;
}

} // namespace sycltla
