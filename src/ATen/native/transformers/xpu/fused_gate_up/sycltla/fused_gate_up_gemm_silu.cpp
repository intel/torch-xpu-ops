/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

#include <limits>

// sycl-tla (CUTLASS for Intel Xe GPU) headers
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/xe_epilogue.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/xe_callbacks.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/fast_math.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

#define SYCL_TLA_CHECK(status)             \
  do {                                     \
    cutlass::Status _err = (status);       \
    TORCH_CHECK(                           \
        _err == cutlass::Status::kSuccess, \
        "sycl-tla error: ",                \
        cutlassGetStatusString(_err),      \
        " at " __FILE__ ":",               \
        __LINE__);                         \
  } while (0)

// -- Tile configurations (TileN=64, M-based dispatch)--
//
// Per-SG tile: 16x32 = 512 accumulators. K=32, PipelineStages=3.
// TileN=64: 344 WGs for N=22016, 95.6% wave efficiency on Arc B580.

using TileShapeSmall = Shape<_32, _64, _32>;
using SGLayoutSmall = Layout<Shape<_2, _2, _1>, Stride<_2, _1, _0>>;

using TileShapeMedium = Shape<_64, _64, _32>;
using SGLayoutMedium = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;

using TileShapeLarge = Shape<_128, _64, _32>;
using SGLayoutLarge = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor; // PyTorch [N,K] weight convention
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using ElementAccumulator = float;

// -- Gate GEMM with SiLU activation epilogue--
//
// Uses LinCombEltAct<SiLu>: D = SiLU(alpha * acc + beta * C)
// With alpha=1, beta=0 this computes D = SiLU(acc).
// GEMM outputs fp16/bf16 directly (v0.7 auto-deduces 16-bit store copy atom).

template <typename Element, typename TileShape_, typename SubGroupLayout_>
static void run_gate_gemm_silu_impl(
    const void* input_ptr,
    const void* weight_ptr,
    void* output_ptr,
    int M,
    int N,
    int K,
    sycl::queue& queue) {
  using MmaAtom = MMA_Atom<XE_DPAS_TT<8, float, Element>>;
  using TiledMma =
      typename TiledMMAHelper<MmaAtom, Layout<TileShape_>, SubGroupLayout_>::
          TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<3>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
  using ElementOutput = Element;

  // LinCombEltAct<SiLu>: D = SiLu(alpha * acc + beta * C)
  using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
      cutlass::epilogue::thread::SiLu,
      ElementOutput,
      float,
      void,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
      EpilogueDispatchPolicy,
      EpilogueOp,
      TileShape_,
      decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape_,
      void, // EpilogueTile (auto-deduce)
      void, // ElementC (no source, beta=0)
      cutlass::gemm::TagToStrideC_t<LayoutC>, // StrideC
      ElementOutput, // ElementD (fp16/bf16)
      cutlass::gemm::TagToStrideC_t<LayoutD>, // StrideD
      FusionCallbacks,
      void, // CopyOpG2R (auto-deduce)
      void>; // CopyOpR2G (auto-deduce)

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape_,
      Element,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      Element,
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

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));

  const Element* A_ptr = reinterpret_cast<const Element*>(input_ptr);
  const Element* B_ptr = reinterpret_cast<const Element*>(weight_ptr);
  Element* D_ptr = reinterpret_cast<Element*>(output_ptr);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {A_ptr, stride_A, B_ptr, stride_B},
      {{ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
       nullptr,
       stride_C,
       D_ptr,
       stride_D},
      hw_info};

  Gemm gemm_op;
  TORCH_CHECK(
      gemm_op.can_implement(arguments) == cutlass::Status::kSuccess,
      "fused_gate_up_silu: gate GEMM+SiLU problem size M=",
      M,
      " N=",
      N,
      " K=",
      K,
      " not supported by compiled tile config");
  SYCL_TLA_CHECK(gemm_op.initialize(arguments, nullptr, &queue));
  SYCL_TLA_CHECK(gemm_op.run(&queue));
}

// -- Up GEMM with multiply epilogue--
//
// EVT epilogue: D = acc * aux_load(silu_buf)
// Uses Sm90EVT<Sm90Compute<multiplies>, Sm90AccFetch, XeAuxLoad>.
// Loads SiLU(gate) result from global memory during epilogue and multiplies
// with up GEMM accumulator, writing the final fused output directly.

template <typename Element, typename TileShape_, typename SubGroupLayout_>
static void run_up_gemm_mul_impl(
    const void* input_ptr,
    const void* weight_ptr,
    const void* aux_ptr,
    void* output_ptr,
    int M,
    int N,
    int K,
    sycl::queue& queue) {
  using MmaAtom = MMA_Atom<XE_DPAS_TT<8, float, Element>>;
  using TiledMma =
      typename TiledMMAHelper<MmaAtom, Layout<TileShape_>, SubGroupLayout_>::
          TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<3>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using ElementOutput = Element;
  using ElementCompute = float;
  constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  // EVT tree: multiplies(AccFetch, AuxLoad)
  using StrideAux = cutlass::gemm::TagToStrideC_t<LayoutD>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies,
          ElementOutput,
          ElementCompute,
          RoundStyle>,
      cutlass::epilogue::fusion::Sm90AccFetch,
      cutlass::epilogue::fusion::XeAuxLoad<Element, StrideAux>>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
      EpilogueDispatchPolicy,
      TileShape_,
      void, // EpilogueTile (auto-deduce)
      void, // ElementC (unused)
      cutlass::gemm::TagToStrideC_t<LayoutC>, // StrideC
      ElementOutput, // ElementD (fp16/bf16)
      cutlass::gemm::TagToStrideC_t<LayoutD>, // StrideD
      EVT,
      void, // CopyOpG2R (auto-deduce)
      void>; // CopyOpR2G (auto-deduce)

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy,
      TileShape_,
      Element,
      cutlass::gemm::TagToStrideA_t<LayoutA>,
      Element,
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

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
  StrideC stride_C =
      cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, N, 1));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));
  StrideAux stride_aux =
      cutlass::make_cute_packed_stride(StrideAux{}, make_shape(M, N, 1));

  const Element* A_ptr = reinterpret_cast<const Element*>(input_ptr);
  const Element* B_ptr = reinterpret_cast<const Element*>(weight_ptr);
  const Element* aux = reinterpret_cast<const Element*>(aux_ptr);
  Element* D_ptr = reinterpret_cast<Element*>(output_ptr);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  // EVT arguments: {AccFetch args, AuxLoad args, Compute args}
  using AuxLoadArgs = typename cutlass::epilogue::fusion::
      XeAuxLoad<Element, StrideAux>::Arguments;
  AuxLoadArgs aux_args;
  aux_args.ptr_aux = aux;
  aux_args.null_default = Element(0);
  aux_args.dAux = stride_aux;

  typename EVT::Arguments evt_args;
  evt_args.op_1 = aux_args;

  typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {A_ptr, stride_A, B_ptr, stride_B},
      {evt_args, nullptr, stride_C, D_ptr, stride_D},
      hw_info};

  Gemm gemm_op;
  TORCH_CHECK(
      gemm_op.can_implement(arguments) == cutlass::Status::kSuccess,
      "fused_gate_up_silu: up GEMM*mul problem size M=",
      M,
      " N=",
      N,
      " K=",
      K,
      " not supported by compiled tile config");
  SYCL_TLA_CHECK(gemm_op.initialize(arguments, nullptr, &queue));
  SYCL_TLA_CHECK(gemm_op.run(&queue));
}

// -- M-based tile dispatch -- gate GEMM + SiLU--

template <typename Element>
static void dispatch_gate_gemm_silu(
    const void* input_ptr,
    const void* weight_ptr,
    void* output_ptr,
    int M,
    int N,
    int K,
    sycl::queue& queue) {
  if (M <= 32) {
    run_gate_gemm_silu_impl<Element, TileShapeSmall, SGLayoutSmall>(
        input_ptr, weight_ptr, output_ptr, M, N, K, queue);
  } else if (M <= 64) {
    run_gate_gemm_silu_impl<Element, TileShapeMedium, SGLayoutMedium>(
        input_ptr, weight_ptr, output_ptr, M, N, K, queue);
  } else {
    run_gate_gemm_silu_impl<Element, TileShapeLarge, SGLayoutLarge>(
        input_ptr, weight_ptr, output_ptr, M, N, K, queue);
  }
}

// -- M-based tile dispatch -- up GEMM * aux_load--

template <typename Element>
static void dispatch_up_gemm_mul(
    const void* input_ptr,
    const void* weight_ptr,
    const void* aux_ptr,
    void* output_ptr,
    int M,
    int N,
    int K,
    sycl::queue& queue) {
  if (M <= 32) {
    run_up_gemm_mul_impl<Element, TileShapeSmall, SGLayoutSmall>(
        input_ptr, weight_ptr, aux_ptr, output_ptr, M, N, K, queue);
  } else if (M <= 64) {
    run_up_gemm_mul_impl<Element, TileShapeMedium, SGLayoutMedium>(
        input_ptr, weight_ptr, aux_ptr, output_ptr, M, N, K, queue);
  } else {
    run_up_gemm_mul_impl<Element, TileShapeLarge, SGLayoutLarge>(
        input_ptr, weight_ptr, aux_ptr, output_ptr, M, N, K, queue);
  }
}

// -- Public API--
//
// Two-kernel EVT pipeline:
//   1. Gate GEMM + SiLU(acc)           -> silu_buf[M, N]  (LinCombEltAct)
//   2. Up GEMM + acc * aux(silu_buf)   -> output[M, N]    (EVT multiplies)
//
// Eliminates the separate SiLU*mul kernel by fusing activation and multiply
// into GEMM epilogues via sycl-tla v0.7 Epilogue Visitor Tree (EVT).
// Halves scratch memory (1x [M,N]) vs the 3-kernel approach (2x [M,N]).

namespace sycltla {

at::Tensor fused_gate_up_silu_sycltla(
    const at::Tensor& input,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight) {
  TORCH_CHECK(input.is_xpu(), "input must be XPU");
  TORCH_CHECK(gate_weight.is_xpu(), "gate_weight must be XPU");
  TORCH_CHECK(up_weight.is_xpu(), "up_weight must be XPU");
  TORCH_CHECK(
      input.device() == gate_weight.device() &&
          input.device() == up_weight.device(),
      "All tensors must be on the same XPU device, got input on ",
      input.device(),
      ", gate_weight on ",
      gate_weight.device(),
      ", up_weight on ",
      up_weight.device());
  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
  TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must be 2D [N, K]");
  TORCH_CHECK(up_weight.dim() == 2, "up_weight must be 2D [N, K]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(gate_weight.is_contiguous(), "gate_weight must be contiguous");
  TORCH_CHECK(up_weight.is_contiguous(), "up_weight must be contiguous");
  TORCH_CHECK(
      input.scalar_type() == gate_weight.scalar_type(),
      "input and gate_weight must have the same dtype, got ",
      input.scalar_type(),
      " and ",
      gate_weight.scalar_type());
  TORCH_CHECK(
      input.scalar_type() == up_weight.scalar_type(),
      "input and up_weight must have the same dtype, got ",
      input.scalar_type(),
      " and ",
      up_weight.scalar_type());
  TORCH_CHECK(input.size(1) == gate_weight.size(1), "K mismatch");
  TORCH_CHECK(gate_weight.size(0) == up_weight.size(0), "N mismatch");

  auto dtype = input.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "Only fp16/bf16 supported, got ",
      dtype);

  const auto M_64 = input.size(0);
  const auto K_64 = input.size(1);
  const auto N_64 = gate_weight.size(0);
  TORCH_CHECK(
      M_64 <= std::numeric_limits<int>::max() &&
          K_64 <= std::numeric_limits<int>::max() &&
          N_64 <= std::numeric_limits<int>::max(),
      "Dimensions M=",
      M_64,
      " K=",
      K_64,
      " N=",
      N_64,
      " exceed int range");
  const int M = static_cast<int>(M_64);
  const int K = static_cast<int>(K_64);
  const int N = static_cast<int>(N_64);

  // Early return for empty shapes to avoid invalid SYCL nd_range submissions
  if (M == 0 || N == 0 || K == 0) {
    return at::empty({M, N}, input.options());
  }

  // silu_buf holds gate GEMM + SiLU result; the up GEMM epilogue reads it
  // via XeAuxLoad and multiplies with up accumulator to produce final output.
  auto silu_buf = at::empty({M, N}, input.options());
  auto output = at::empty({M, N}, input.options());

  auto& queue = at::xpu::getCurrentXPUStream(input.device().index()).queue();

  if (dtype == at::kHalf) {
    // Step 1: gate GEMM + SiLU -> silu_buf
    dispatch_gate_gemm_silu<cutlass::half_t>(
        input.data_ptr(),
        gate_weight.data_ptr(),
        silu_buf.data_ptr(),
        M,
        N,
        K,
        queue);
    // Step 2: up GEMM * silu_buf -> output
    dispatch_up_gemm_mul<cutlass::half_t>(
        input.data_ptr(),
        up_weight.data_ptr(),
        silu_buf.data_ptr(),
        output.data_ptr(),
        M,
        N,
        K,
        queue);
  } else {
    // Step 1: gate GEMM + SiLU -> silu_buf
    dispatch_gate_gemm_silu<cutlass::bfloat16_t>(
        input.data_ptr(),
        gate_weight.data_ptr(),
        silu_buf.data_ptr(),
        M,
        N,
        K,
        queue);
    // Step 2: up GEMM * silu_buf -> output
    dispatch_up_gemm_mul<cutlass::bfloat16_t>(
        input.data_ptr(),
        up_weight.data_ptr(),
        silu_buf.data_ptr(),
        output.data_ptr(),
        M,
        N,
        K,
        queue);
  }

  return output;
}

} // namespace sycltla
