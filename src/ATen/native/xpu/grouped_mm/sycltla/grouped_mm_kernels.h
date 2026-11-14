/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Interface between the host-side grouped_mm glue (grouped_mm_api.cpp, built
// into torch_xpu_ops) and the SYCL-TLA kernels (built into the
// torch-xpu-ops-sycltla library). Deliberately free of CUTLASS/CuTe includes
// so that the host side compiles without SYCL-TLA headers.

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

namespace sycltla {
namespace grouped_gemm {

enum class DType : int { BF16 = 0, FP16 = 1 };

// All sizes/strides are in elements. Groups are described by `offs`, the
// PyTorch int32 cumulative row offsets (offs[g] is the exclusive end row of
// group g; group 0 starts at row 0).
struct Params {
  const void* a = nullptr;
  const void* b = nullptr;
  void* d = nullptr;
  const int32_t* offs = nullptr;
  int32_t num_groups = 0;
  // M-grouped: unused. K-grouped: the fixed M of every group's output.
  int32_t M = 0;
  int32_t N = 0;
  // M-grouped: the fixed contraction length. K-grouped: unused (per-group).
  int32_t K = 0;
  int64_t lda = 0;
  int64_t ldb = 0;
  int64_t ldd = 0;
  // M-grouped: distance between consecutive B matrices (B is [E, K, N]).
  int64_t group_stride_b = 0;
  // K-grouped: distance between consecutive D matrices (D is [E, M, N]).
  int64_t group_stride_d = 0;
  // Persistent grid size (work-groups). 0 => default_num_workgroups().
  int32_t num_workgroups = 0;
  // Work-group tile, one of the kTile* values below. kTileAuto =>
  // choose_tile().
  int32_t tile = 0;
};

// Work-group tile configurations (Params::tile).
constexpr int32_t kTileAuto = 0;
constexpr int32_t kTile256x128 = 1; // 16 sub-groups
constexpr int32_t kTile256x256 = 2; // 32 sub-groups, preferred on PVC
constexpr int32_t kTile128x128 = 3; // 8 sub-groups, for small groups
constexpr int32_t kTile128x256 = 4; // 16 sub-groups, for small groups

// M-grouped GEMM (forward / grad_input of torch._grouped_mm, 2D x 3D case):
//   D[rows_g, :] = A[rows_g, :K] @ B_g[K, N]        for every group g
// A is row-major with leading dimension lda, D is row-major with ldd.
// B_g is addressed as an (N, K) matrix: LayoutB == 'R' means element (n, k)
// lives at n * ldb + k (i.e. the [K, N] operand is K-contiguous / transposed
// weights); LayoutB == 'C' means (n, k) lives at n + k * ldb (the usual
// row-major [K, N] weight).
template <DType dt, char LayoutB>
void run_m_grouped(sycl::queue& queue, const Params& p);

// K-grouped GEMM (grad_weight of torch._grouped_mm, 2D x 2D case):
//   D_g[M, N] = A[:, rows_g] @ B[rows_g, :]            for every group g
// A is an (M, R) matrix stored M-contiguous (element (m, r) at m + r * lda),
// i.e. the transposed view of a row-major [R, M] activation. B is row-major
// [R, N] with leading dimension ldb. D_g is row-major [M, N] with ldd.
template <DType dt>
void run_k_grouped(sycl::queue& queue, const Params& p);

// Tile for a call: p.tile if set, else default_tile(). Never depends on the
// routed row counts, so all ranks of an expert-parallel job agree.
int choose_tile(const sycl::device& dev, const Params& p);

// Device default tile when nothing is known about the call.
int default_tile(const sycl::device& dev);

// Persistent work-groups to launch on `dev` for `tile`: enough to fill every
// Xe core (512 threads per core in GRF-256 mode).
int default_num_workgroups(const sycl::device& dev, int tile);

// Number of Xe cores on `dev`.
int xe_core_count(const sycl::device& dev);

// "256x128", "256x256", "128x128", "128x256".
const char* tile_name(int tile);

} // namespace grouped_gemm
} // namespace sycltla
