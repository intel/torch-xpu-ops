// Copyright 2020-2026 Intel Corporation
// Licensed under the Apache License, Version 2.0

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <sycl/sycl.hpp>

// sycl-tla headers (same as research kernel)
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/xe_epilogue.hpp>
#include <cutlass/epilogue/fusion/xe_callbacks.hpp>
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

// ── Tile configurations (TileN=64, M-based dispatch) ────────────────────────
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

// ── GEMM template ────────────────────────────────────────────────────────────
//
// GEMM outputs fp16/bf16 directly (v0.7 auto-deduces 16-bit store copy atom).
// Beta=0, no C source load needed.  Combined buffer is fp16/bf16, halving
// scratch memory and bandwidth vs float32.

template <typename Element, typename TileShape_, typename SubGroupLayout_>
static void run_sycl_tla_gemm_impl(
    const void* input_ptr,
    const void* weight_ptr,
    void* output_ptr,
    int M,
    int N,
    int K,
    int ldd,
    sycl::queue& queue) {
  using MmaAtom = MMA_Atom<XE_DPAS_TT<8, float, Element>>;
  using TiledMma =
      typename TiledMMAHelper<MmaAtom, Layout<TileShape_>, SubGroupLayout_>::
          TiledMMA;

  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<3>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  // 16-bit output: v0.7 auto-deduces correct store copy atom for ElementD
  // bitwidth.
  using ElementOutput = Element;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
      ElementOutput,
      float,
      ElementAccumulator,
      ElementAccumulator,
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
      void, // CopyOpG2R (auto-deduce, no C load)
      void>; // CopyOpR2G (auto-deduce → 16-bit store)

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
      cutlass::make_cute_packed_stride(StrideC{}, make_shape(M, ldd, 1));
  StrideD stride_D =
      cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, ldd, 1));

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
      "fused_gate_up_silu: problem size M=",
      M,
      " N=",
      N,
      " K=",
      K,
      " not supported by compiled tile config");
  SYCL_TLA_CHECK(gemm_op.initialize(arguments, nullptr, &queue));
  SYCL_TLA_CHECK(gemm_op.run(&queue));
}

// ── M-based tile dispatch ───────────────────────────────────────────────────

template <typename Element>
static void dispatch_sycl_tla_gemm(
    const void* input_ptr,
    const void* weight_ptr,
    void* output_ptr,
    int M,
    int N,
    int K,
    int ldd,
    sycl::queue& queue) {
  if (M <= 32) {
    run_sycl_tla_gemm_impl<Element, TileShapeSmall, SGLayoutSmall>(
        input_ptr, weight_ptr, output_ptr, M, N, K, ldd, queue);
  } else if (M <= 64) {
    run_sycl_tla_gemm_impl<Element, TileShapeMedium, SGLayoutMedium>(
        input_ptr, weight_ptr, output_ptr, M, N, K, ldd, queue);
  } else {
    run_sycl_tla_gemm_impl<Element, TileShapeLarge, SGLayoutLarge>(
        input_ptr, weight_ptr, output_ptr, M, N, K, ldd, queue);
  }
}

// ── SiLU * mul kernel ───────────────────────────────────────────────────────
//
// Reads 16-bit GEMM output (combined), applies SiLU*mul in fp32, writes
// fp16/bf16.  2D nd_range eliminates integer division (N=11008 not pow2).

template <typename Element>
struct SiluMulKernel {};

template <typename Element>
static void launch_silu_mul_impl(
    const void* combined_ptr,
    void* output_ptr,
    int M,
    int N,
    sycl::queue& queue) {
  const Element* in = reinterpret_cast<const Element*>(combined_ptr);
  Element* out = reinterpret_cast<Element*>(output_ptr);

  constexpr int WG_N = 256;
  int N_padded = ((N + WG_N - 1) / WG_N) * WG_N;

  queue.parallel_for<SiluMulKernel<Element>>(
      sycl::nd_range<2>(sycl::range<2>(M, N_padded), sycl::range<2>(1, WG_N)),
      [=](sycl::nd_item<2> item) {
        int row = item.get_global_id(0);
        int col = item.get_global_id(1);
        if (col >= N)
          return;
        int base = row * 2 * N + col;
        float gate = static_cast<float>(in[base]);
        float up = static_cast<float>(in[base + N]);
        float silu = gate / (1.0f + sycl::exp(-gate));
        out[row * N + col] = static_cast<Element>(silu * up);
      });
}

// ── Public API ──────────────────────────────────────────────────────────────

namespace sycltla {

at::Tensor fused_gate_up_silu_sycltla(
    const at::Tensor& input,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight) {
  TORCH_CHECK(input.is_xpu(), "input must be XPU");
  TORCH_CHECK(gate_weight.is_xpu(), "gate_weight must be XPU");
  TORCH_CHECK(up_weight.is_xpu(), "up_weight must be XPU");
  TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");
  TORCH_CHECK(gate_weight.dim() == 2, "gate_weight must be 2D [N, K]");
  TORCH_CHECK(up_weight.dim() == 2, "up_weight must be 2D [N, K]");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(gate_weight.is_contiguous(), "gate_weight must be contiguous");
  TORCH_CHECK(up_weight.is_contiguous(), "up_weight must be contiguous");
  TORCH_CHECK(input.scalar_type() == gate_weight.scalar_type());
  TORCH_CHECK(input.scalar_type() == up_weight.scalar_type());
  TORCH_CHECK(input.size(1) == gate_weight.size(1), "K mismatch");
  TORCH_CHECK(gate_weight.size(0) == up_weight.size(0), "N mismatch");

  auto dtype = input.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "Only fp16/bf16 supported, got ",
      dtype);

  const int M = input.size(0);
  const int K = input.size(1);
  const int N = gate_weight.size(0);

  // GEMM outputs fp16/bf16 directly; combined buffer matches input dtype.
  // This halves scratch memory vs the previous float32 approach.
  auto combined = at::empty({M, 2 * N}, input.options());
  auto output = at::empty({M, N}, input.options()); // result (fp16/bf16)

  auto& queue = at::xpu::getCurrentXPUStream(input.device().index()).queue();
  void* combined_ptr = combined.data_ptr();

  // Two GEMMs into combined [M, 2N] with ldd=2*N (no weight concatenation)
  if (dtype == at::kHalf) {
    auto* ptr = reinterpret_cast<cutlass::half_t*>(combined_ptr);
    dispatch_sycl_tla_gemm<cutlass::half_t>(
        input.data_ptr(),
        gate_weight.data_ptr(),
        ptr,
        M,
        N,
        K,
        2 * N,
        queue); // gate -> cols [0, N)
    dispatch_sycl_tla_gemm<cutlass::half_t>(
        input.data_ptr(),
        up_weight.data_ptr(),
        ptr + N,
        M,
        N,
        K,
        2 * N,
        queue); // up -> cols [N, 2N)
    launch_silu_mul_impl<cutlass::half_t>(
        combined_ptr, output.data_ptr(), M, N, queue);
  } else {
    auto* ptr = reinterpret_cast<cutlass::bfloat16_t*>(combined_ptr);
    dispatch_sycl_tla_gemm<cutlass::bfloat16_t>(
        input.data_ptr(), gate_weight.data_ptr(), ptr, M, N, K, 2 * N, queue);
    dispatch_sycl_tla_gemm<cutlass::bfloat16_t>(
        input.data_ptr(), up_weight.data_ptr(), ptr + N, M, N, K, 2 * N, queue);
    launch_silu_mul_impl<cutlass::bfloat16_t>(
        combined_ptr, output.data_ptr(), M, N, queue);
  }

  return output;
}

} // namespace sycltla