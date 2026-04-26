/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * SYCL Grouped Matrix Multiplication kernel using sycl-tla (CUTLASS for Intel
 * GPUs). Supports BF16 inputs with FP32 accumulation and BF16 output. Handles
 * all 4 input modes: 3D×3D, 2D×3D, 3D×2D, 2D×2D.
 *
 * Reference: sycl-tla/examples/04_bmg_grouped_gemm/04_bmg_grouped_gemm.cpp
 */

#include <ATen/ATen.h>
#include <ATen/native/xpu/sycltla/GroupedMM.h>

#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/collective/xe_array_epilogue.hpp>
#include <cutlass/epilogue/fusion/xe_callbacks.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>

#include <cute/tensor.hpp>

#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>

#include <vector>

using namespace cute;

namespace {

// ---------------------------------------------------------------------------
// Type configuration
// ---------------------------------------------------------------------------
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using ElementA = bfloat16_t;
using ElementB = bfloat16_t;
using ElementOutput = bfloat16_t;
using ElementAccumulator = float;
using ElementComputeEpi = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// ---------------------------------------------------------------------------
// Kernel type assembly
// ---------------------------------------------------------------------------
using GmemTiledCopyA = void;
using GmemTiledCopyB = void;

using TileShape = Shape<_256, _256, _32>;

using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, ElementAccumulator, ElementA>>,
    Layout<TileShape>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

constexpr int PipelineStages = 2;

using GEMMDispatchPolicy =
    cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput,
    ElementComputeEpi,
    ElementAccumulator,
    ElementAccumulator,
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
    ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC*>,
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD*>,
    FusionCallBacks,
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

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

// ---------------------------------------------------------------------------
// run_grouped_gemm — launch the grouped GEMM kernel
// ---------------------------------------------------------------------------
cutlass::Status run_grouped_gemm(
    int group_count,
    const std::vector<typename ProblemShape::UnderlyingProblemShape>&
        problem_sizes_host,
    const std::vector<const ElementA*>& ptr_a_host,
    const std::vector<const ElementB*>& ptr_b_host,
    const std::vector<ElementOutput*>& ptr_d_host,
    const std::vector<StrideA>& stride_a_host,
    const std::vector<StrideB>& stride_b_host,
    const std::vector<StrideD>& stride_d_host) {
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
      problem_sizes_device;
  problem_sizes_device.reset(group_count);
  problem_sizes_device.copy_from_host(problem_sizes_host.data());

  cutlass::DeviceAllocation<const ElementA*> ptr_A_device;
  ptr_A_device.reset(group_count);
  ptr_A_device.copy_from_host(ptr_a_host.data());

  cutlass::DeviceAllocation<const ElementB*> ptr_B_device;
  ptr_B_device.reset(group_count);
  ptr_B_device.copy_from_host(ptr_b_host.data());

  // C is unused (beta=0); pass nullptr to avoid type mismatch UB.
  cutlass::DeviceAllocation<const ElementAccumulator*> ptr_C_device;
  ptr_C_device.reset(group_count);
  std::vector<const ElementAccumulator*> ptr_c_host(group_count, nullptr);
  ptr_C_device.copy_from_host(ptr_c_host.data());

  cutlass::DeviceAllocation<ElementOutput*> ptr_D_device;
  ptr_D_device.reset(group_count);
  ptr_D_device.copy_from_host(ptr_d_host.data());

  cutlass::DeviceAllocation<StrideA> stride_A_device;
  stride_A_device.reset(group_count);
  stride_A_device.copy_from_host(stride_a_host.data());

  cutlass::DeviceAllocation<StrideB> stride_B_device;
  stride_B_device.reset(group_count);
  stride_B_device.copy_from_host(stride_b_host.data());

  cutlass::DeviceAllocation<StrideC> stride_C_device;
  stride_C_device.reset(group_count);
  stride_C_device.copy_from_host(stride_d_host.data());

  cutlass::DeviceAllocation<StrideD> stride_D_device;
  stride_D_device.reset(group_count);
  stride_D_device.copy_from_host(stride_d_host.data());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  using RasterOrderOptions =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<
          ProblemShape>::RasterOrderOptions;

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

  arguments = typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_sizes_device.get(), problem_sizes_host.data()},
      {ptr_A_device.get(),
       stride_A_device.get(),
       ptr_B_device.get(),
       stride_B_device.get()},
      {fusion_args,
       ptr_C_device.get(),
       stride_C_device.get(),
       ptr_D_device.get(),
       stride_D_device.get()},
      hw_info,
      {1, RasterOrderOptions::AlongN}};

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  cutlass::Status status;

  status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  status = gemm_op.run();
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  compat::wait();
  return cutlass::Status::kSuccess;
}

} // anonymous namespace

namespace at::xpu::detail {

void bf16bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias,
    at::Tensor& out) {
  TORCH_CHECK(
      out.is_contiguous(),
      "grouped_mm: output tensor must be contiguous");
  TORCH_CHECK(
      !bias.has_value(),
      "grouped_mm: bias is not supported for sycl-tla grouped_mm");

  mat_a = mat_a.contiguous();
  mat_b = mat_b.contiguous();

  bool a_is_2d = mat_a.dim() == 2;
  bool b_is_2d = mat_b.dim() == 2;

  int group_count;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  std::vector<const ElementA*> ptr_a_vec;
  std::vector<const ElementB*> ptr_b_vec;
  std::vector<ElementOutput*> ptr_d_vec;
  std::vector<StrideA> stride_a_vec;
  std::vector<StrideB> stride_b_vec;
  std::vector<StrideD> stride_d_vec;

  auto* base_a = reinterpret_cast<const ElementA*>(mat_a.data_ptr());
  auto* base_b = reinterpret_cast<const ElementB*>(mat_b.data_ptr());
  auto* base_d = reinterpret_cast<ElementOutput*>(out.data_ptr());

  // Read offs tensor to host
  std::vector<int32_t> offs_host;
  if (offs.has_value()) {
    auto offs_cpu = offs->cpu().contiguous();
    const int32_t* p = offs_cpu.data_ptr<int32_t>();
    offs_host.assign(p, p + offs_cpu.numel());
  }

  if (!a_is_2d && !b_is_2d) {
    // 3D x 3D: regular batched MM
    group_count = mat_a.size(0);
    int M = mat_a.size(1);
    int N = mat_b.size(2);
    int K = mat_a.size(2);

    for (int g = 0; g < group_count; ++g) {
      problem_sizes.push_back({M, N, K});
      ptr_a_vec.push_back(base_a + g * M * K);
      ptr_b_vec.push_back(base_b + g * K * N);
      ptr_d_vec.push_back(base_d + g * M * N);
      stride_a_vec.push_back(
          cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
      stride_b_vec.push_back(
          cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
      stride_d_vec.push_back(
          cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    }
  } else if (a_is_2d && !b_is_2d) {
    // 2D x 3D: ragged A (MoE pattern)
    TORCH_CHECK(
        offs.has_value(),
        "grouped_mm: 2D x 3D mode requires offs tensor");
    group_count = mat_b.size(0);
    TORCH_CHECK(
        static_cast<int>(offs_host.size()) == group_count,
        "grouped_mm: offs length (", offs_host.size(),
        ") must match group count (", group_count, ")");
    int K = mat_a.size(1);
    int N = mat_b.size(2);
    int64_t out_stride_row = out.size(1);

    int32_t row_start = 0;
    for (int g = 0; g < group_count; ++g) {
      int32_t row_end = offs_host[g];
      int M_g = row_end - row_start;

      problem_sizes.push_back({M_g, N, K});
      ptr_a_vec.push_back(base_a + row_start * K);
      ptr_b_vec.push_back(base_b + g * K * N);
      ptr_d_vec.push_back(base_d + row_start * out_stride_row);
      stride_a_vec.push_back(
          cutlass::make_cute_packed_stride(StrideA{}, {M_g, K, 1}));
      stride_b_vec.push_back(
          cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
      stride_d_vec.push_back(
          cutlass::make_cute_packed_stride(StrideD{}, {M_g, N, 1}));

      row_start = row_end;
    }
  } else if (!a_is_2d && b_is_2d) {
    // 3D x 2D: ragged B
    TORCH_CHECK(
        offs.has_value(),
        "grouped_mm: 3D x 2D mode requires offs tensor");
    group_count = mat_a.size(0);
    TORCH_CHECK(
        static_cast<int>(offs_host.size()) == group_count,
        "grouped_mm: offs length (", offs_host.size(),
        ") must match group count (", group_count, ")");
    int M = mat_a.size(1);
    int K = mat_a.size(2);

    std::vector<at::Tensor> b_slices;
    std::vector<at::Tensor> d_slices;

    int32_t col_start = 0;
    for (int g = 0; g < group_count; ++g) {
      int32_t col_end = offs_host[g];
      int N_g = col_end - col_start;

      auto b_slice = mat_b.slice(1, col_start, col_end).contiguous();
      auto d_slice = at::empty({M, N_g}, mat_a.options());
      b_slices.push_back(b_slice);
      d_slices.push_back(d_slice);

      problem_sizes.push_back({M, N_g, K});
      ptr_a_vec.push_back(base_a + g * M * K);
      ptr_b_vec.push_back(
          reinterpret_cast<const ElementB*>(b_slice.data_ptr()));
      ptr_d_vec.push_back(reinterpret_cast<ElementOutput*>(d_slice.data_ptr()));

      stride_a_vec.push_back(
          cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
      stride_b_vec.push_back(
          cutlass::make_cute_packed_stride(StrideB{}, {N_g, K, 1}));
      stride_d_vec.push_back(
          cutlass::make_cute_packed_stride(StrideD{}, {M, N_g, 1}));

      col_start = col_end;
    }

    cutlass::Status status = run_grouped_gemm(
        group_count,
        problem_sizes,
        ptr_a_vec,
        ptr_b_vec,
        ptr_d_vec,
        stride_a_vec,
        stride_b_vec,
        stride_d_vec);

    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "SYCL grouped_mm kernel failed with status ",
        int(status));

    col_start = 0;
    for (int g = 0; g < group_count; ++g) {
      int32_t col_end = offs_host[g];
      out.slice(1, col_start, col_end).copy_(d_slices[g]);
      col_start = col_end;
    }
    return;
  } else {
    // 2D x 2D: ragged K
    TORCH_CHECK(
        offs.has_value(),
        "grouped_mm: 2D x 2D mode requires offs tensor");
    group_count = offs_host.size();
    int M = mat_a.size(0);
    int N = mat_b.size(1);

    std::vector<at::Tensor> a_slices;

    int32_t k_start = 0;
    for (int g = 0; g < group_count; ++g) {
      int32_t k_end = offs_host[g];
      int K_g = k_end - k_start;

      auto a_slice = mat_a.slice(1, k_start, k_end).contiguous();
      a_slices.push_back(a_slice);

      problem_sizes.push_back({M, N, K_g});
      ptr_a_vec.push_back(
          reinterpret_cast<const ElementA*>(a_slice.data_ptr()));
      ptr_b_vec.push_back(base_b + k_start * N);
      ptr_d_vec.push_back(base_d + g * M * N);

      stride_a_vec.push_back(
          cutlass::make_cute_packed_stride(StrideA{}, {M, K_g, 1}));
      stride_b_vec.push_back(
          cutlass::make_cute_packed_stride(StrideB{}, {N, K_g, 1}));
      stride_d_vec.push_back(
          cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));

      k_start = k_end;
    }
  }

  cutlass::Status status = run_grouped_gemm(
      group_count,
      problem_sizes,
      ptr_a_vec,
      ptr_b_vec,
      ptr_d_vec,
      stride_a_vec,
      stride_b_vec,
      stride_d_vec);

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "SYCL grouped_mm kernel failed with status ",
      int(status));
}

} // namespace at::xpu::detail
