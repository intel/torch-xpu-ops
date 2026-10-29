/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions derived from SYCL-TLA
 * examples/12_xe20_moe_gemm_cute_interface (BSD-3-Clause):
 * Copyright (c) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Persistent grouped GEMM for Intel Xe GPUs written against the CuTe
// "cute interface" of SYCL-TLA. One work-group of 16 sub-groups owns a
// 256x128 output tile at a time and walks a linearized tile space that spans
// all groups; group boundaries come from PyTorch's cumulative `offs` tensor so
// no host synchronization is needed. Two schedulings share one tile GEMM:
//   * M-grouped: groups partition the rows of A and D (fwd / grad_input).
//   * K-grouped: groups partition the contraction dimension (grad_weight).
// This header only contains templates; see grouped_mm_*.cpp for the explicit
// instantiations that split compilation across translation units.

#pragma once

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>

#include <ATen/native/xpu/grouped_mm/sycltla/grouped_mm_kernels.h>

#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

namespace sycltla {
namespace grouped_gemm {

using namespace cute;

template <DType dt>
struct CuteElem;
template <>
struct CuteElem<DType::BF16> {
  using type = cute::bfloat16_t;
};
template <>
struct CuteElem<DType::FP16> {
  using type = cute::half_t;
};

// A tile config ("Cfg") provides:
//   WGTile    Shape<_M, _N, _K>   work-group tile
//   SGLayout  Layout<...>         sub-group arrangement over the tile
//   kTileM / kTileN / kTileK      the same numbers as ints
//   kPrefetchDist                 K tiles prefetched ahead of the mainloop
// The production instantiations use the four configs below; the standalone
// tuner (test/microbench/sycltla_grouped_mm_tune) instantiates more.
//
// Work-group tile configuration: 256x128x32 tile, 8x2 sub-groups (n-major),
// each sub-group owning a 32x64 slab of the accumulator.
struct TileCfg256x128 {
  using WGTile = Shape<_256, _128, _32>;
  using SGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
  static constexpr int kTileM = 256;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 32;
  static constexpr int kPrefetchDist = 3;
};

// 256x256x32 tile, 8x4 sub-groups (32 sub-groups = 512 threads), each
// sub-group again owning 32x64 of the accumulator. Preferred on PVC.
struct TileCfg256x256 {
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  static constexpr int kTileM = 256;
  static constexpr int kTileN = 256;
  static constexpr int kTileK = 32;
  static constexpr int kPrefetchDist = 3;
};

// 128x128x32 tile, 4x2 sub-groups (8 sub-groups = 128 threads), 32x64 per
// sub-group. Small groups: less padding in a group's last row tile, and four
// work-groups per Xe core.
struct TileCfg128x128 {
  using WGTile = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 128;
  static constexpr int kTileK = 32;
  static constexpr int kPrefetchDist = 3;
};

// 128x256x32 tile, 4x4 sub-groups (16 sub-groups = 256 threads), 32x64 per
// sub-group. Small groups: half the row padding of the 256-row tiles, while
// keeping a 256-wide N tile for B reuse.
struct TileCfg128x256 {
  using WGTile = Shape<_128, _256, _32>;
  using SGLayout = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  static constexpr int kTileM = 128;
  static constexpr int kTileN = 256;
  static constexpr int kTileK = 32;
  static constexpr int kPrefetchDist = 3;
};

template <class T, class Cfg>
using TiledMmaFor = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, T, T>>,
    Layout<typename Cfg::WGTile>,
    typename Cfg::SGLayout>::TiledMMA;

// (rows, cols) matrix view. 'R': element (r, c) at r * ld + c.
//                           'C': element (r, c) at r + c * ld.
template <char LayoutKind, class T>
CUTE_DEVICE auto make_2d_tensor(T* ptr, int rows, int cols, int ld) {
  auto shape = make_shape(rows, cols);
  if constexpr (LayoutKind == 'C') {
    return make_tensor(
        make_gmem_ptr<T>(ptr), make_layout(shape, make_stride(_1{}, ld)));
  } else {
    return make_tensor(
        make_gmem_ptr<T>(ptr), make_layout(shape, make_stride(ld, _1{})));
  }
}

// One (TileM x TileN) output tile: D(m_idx, n_idx) = A(m_idx, :) * B(n_idx,
// :)^T A is (M, K), B is (N, K), D is (M, N). Loads/stores are Xe 2D block
// messages chosen automatically from the tensor strides (regular, VNNI or
// transposed), so any of the operands may be row- or column-major. Out-of-
// bounds rows/columns are zero-filled on load and masked on store, which is
// what makes ragged group sizes work without padding.
template <
    int PrefetchDist,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA>
CUTE_DEVICE void gemm_tile(
    ATensor const& A,
    BTensor const& B,
    DTensor& D,
    int m_idx,
    int n_idx,
    TiledMMA const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  auto local_id = item.get_local_linear_id();

  Tensor cA = make_identity_tensor(A.shape()); // (M,K)
  Tensor cB = make_identity_tensor(B.shape()); // (N,K)
  Tensor cD = make_identity_tensor(D.shape()); // (M,N)

  auto wg_coord = make_coord(m_idx, n_idx, 0);
  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(m_idx, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(n_idx, _));
  Tensor gD = local_tile(cD, wg_tile, wg_coord, Step<_1, _1, X>{});

  auto thr_mma = mma.get_slice(local_id);

  auto tiled_copy_a = get_block_2d_copy_A<void>(mma, A);
  auto tiled_copy_b = get_block_2d_copy_B<void>(mma, B);
  auto tiled_copy_d = get_block_2d_copy_D<void>(mma, D);

  auto thr_copy_a = tiled_copy_a.get_slice(local_id);
  auto thr_copy_b = tiled_copy_b.get_slice(local_id);
  auto thr_copy_d = tiled_copy_d.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_sg_fragment_C(gD);
  auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  auto tCgD = thr_copy_d.partition_D(gD);

  auto prefetch_a = make_block_2d_prefetch(tiled_copy_a);
  auto prefetch_b = make_block_2d_prefetch(tiled_copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  constexpr int barrier_scope = 2;
  constexpr int prefetch_dist = PrefetchDist;
  const int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int prefetch_k = 0;

  clear(tCrD);

  // Bounded by k_tile_count: with an empty contraction (K == 0, e.g. an
  // empty group in the K-grouped kernel) there is nothing to prefetch.
  CUTE_UNROLL
  for (; prefetch_k < prefetch_dist && prefetch_k < k_tile_count;
       prefetch_k++) {
    prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
    prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
    barrier_arrive(barrier_scope);

    copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);

    if (prefetch_k < k_tile_count) {
      prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
      prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    cute::gemm(mma, tCrA, tCrB, tCrD);
    barrier_wait(barrier_scope);
  }

  reorder(tCrD, tCrD_final);
  copy(tiled_copy_d, tCrD_final, tCgD);
}

// Linearized, device-side tile scheduler over all groups. Every work-group
// walks the same group sequence, so the per-group walk is amortized across
// the persistent loop (tile indices handed to one work-group only increase).
struct GroupCursor {
  int32_t g = 0;
  int32_t row_start = 0;
  int32_t rows = 0;
  int32_t tiles_m = 0;
  int64_t tile_start = 0;
  int64_t tiles = 0;
};

template <bool KGrouped, int TileM, int TileN>
struct Scheduler {
  const int32_t* offs;
  int32_t num_groups;
  int32_t fixed_m; // K-grouped only
  int32_t n;

  CUTE_DEVICE void describe(GroupCursor& c) const {
    int32_t start = (c.g == 0) ? 0 : offs[c.g - 1];
    int32_t end = offs[c.g];
    start = start < 0 ? 0 : start;
    c.row_start = start;
    c.rows = (end > start) ? (end - start) : 0;
    const int32_t m = KGrouped ? fixed_m : c.rows;
    c.tiles_m = (m + TileM - 1) / TileM;
    c.tiles = int64_t(c.tiles_m) * int64_t((n + TileN - 1) / TileN);
  }

  // Resolve linear tile index -> (group, m_idx, n_idx). Returns false once
  // the index lies past the last group.
  CUTE_DEVICE bool locate(
      int64_t linear,
      GroupCursor& c,
      int32_t& m_idx,
      int32_t& n_idx) const {
    while (c.tile_start + c.tiles <= linear) {
      c.tile_start += c.tiles;
      ++c.g;
      if (c.g >= num_groups) {
        return false;
      }
      describe(c);
    }
    const int64_t local = linear - c.tile_start;
    m_idx = int32_t(local % c.tiles_m);
    n_idx = int32_t(local / c.tiles_m);
    return true;
  }
};

template <class T, char LayoutB, class Cfg>
struct MGroupedKernel {
  Params p;

  void operator()(sycl::nd_item<3> item) const {
    const auto mma = TiledMmaFor<T, Cfg>{};
    const Scheduler<false, Cfg::kTileM, Cfg::kTileN> sched{
        p.offs, p.num_groups, 0, p.N};
    GroupCursor cur;
    sched.describe(cur);

    const int64_t wg = item.get_group_linear_id();
    const int64_t nwg = int64_t(item.get_group_range(0)) *
        item.get_group_range(1) * item.get_group_range(2);

    T* a = const_cast<T*>(static_cast<const T*>(p.a));
    T* b = const_cast<T*>(static_cast<const T*>(p.b));
    T* d = static_cast<T*>(p.d);

    for (int64_t lin = wg;; lin += nwg) {
      int32_t m_idx, n_idx;
      if (!sched.locate(lin, cur, m_idx, n_idx)) {
        break;
      }
      auto A = make_2d_tensor<'R'>(
          a + int64_t(cur.row_start) * p.lda, cur.rows, p.K, int(p.lda));
      auto B = make_2d_tensor<LayoutB>(
          b + int64_t(cur.g) * p.group_stride_b, p.N, p.K, int(p.ldb));
      auto D = make_2d_tensor<'R'>(
          d + int64_t(cur.row_start) * p.ldd, cur.rows, p.N, int(p.ldd));
      gemm_tile<Cfg::kPrefetchDist>(A, B, D, m_idx, n_idx, mma);
    }
  }
};

template <class T, class Cfg>
struct KGroupedKernel {
  Params p;

  void operator()(sycl::nd_item<3> item) const {
    const auto mma = TiledMmaFor<T, Cfg>{};
    const Scheduler<true, Cfg::kTileM, Cfg::kTileN> sched{
        p.offs, p.num_groups, p.M, p.N};
    GroupCursor cur;
    sched.describe(cur);

    const int64_t wg = item.get_group_linear_id();
    const int64_t nwg = int64_t(item.get_group_range(0)) *
        item.get_group_range(1) * item.get_group_range(2);

    T* a = const_cast<T*>(static_cast<const T*>(p.a));
    T* b = const_cast<T*>(static_cast<const T*>(p.b));
    T* d = static_cast<T*>(p.d);

    for (int64_t lin = wg;; lin += nwg) {
      int32_t m_idx, n_idx;
      if (!sched.locate(lin, cur, m_idx, n_idx)) {
        break;
      }
      // A: (M, rows) with element (m, r) at m + r * lda  -> column-major.
      auto A = make_2d_tensor<'C'>(
          a + int64_t(cur.row_start) * p.lda, p.M, cur.rows, int(p.lda));
      // B as (N, rows) with element (n, r) at n + r * ldb -> column-major
      // view of the row-major [rows, N] operand.
      auto B = make_2d_tensor<'C'>(
          b + int64_t(cur.row_start) * p.ldb, p.N, cur.rows, int(p.ldb));
      auto D = make_2d_tensor<'R'>(
          d + int64_t(cur.g) * p.group_stride_d, p.M, p.N, int(p.ldd));
      gemm_tile<Cfg::kPrefetchDist>(A, B, D, m_idx, n_idx, mma);
    }
  }
};

template <class KernelT>
void launch(
    sycl::queue& queue,
    const KernelT& kernel,
    int num_workgroups,
    int threads_per_workgroup) {
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;
  syclex::properties kernel_props{
      syclex::sub_group_size<16>, intelex::grf_size<256>};
  const size_t local = size_t(threads_per_workgroup);
  sycl::range<3> local_range{1, 1, local};
  sycl::range<3> global_range{1, 1, local * size_t(num_workgroups)};
  queue.parallel_for<KernelT>(
      sycl::nd_range<3>(global_range, local_range), kernel_props, kernel);
}

template <class Cfg, DType dt, char LayoutB>
void run_m_grouped_cfg(sycl::queue& queue, const Params& p, int num_wgs) {
  using T = typename CuteElem<dt>::type;
  MGroupedKernel<T, LayoutB, Cfg> kernel{p};
  launch(queue, kernel, num_wgs, int(size(TiledMmaFor<T, Cfg>{})));
}

template <class Cfg, DType dt>
void run_k_grouped_cfg(sycl::queue& queue, const Params& p, int num_wgs) {
  using T = typename CuteElem<dt>::type;
  KGroupedKernel<T, Cfg> kernel{p};
  launch(queue, kernel, num_wgs, int(size(TiledMmaFor<T, Cfg>{})));
}

template <DType dt, char LayoutB>
void run_m_grouped(sycl::queue& queue, const Params& p) {
  if (p.num_groups <= 0 || p.N <= 0) {
    return;
  }
  const int tile = choose_tile(queue.get_device(), p);
  const int num_wgs = p.num_workgroups > 0
      ? p.num_workgroups
      : default_num_workgroups(queue.get_device(), tile);
  if (tile == kTile256x256) {
    run_m_grouped_cfg<TileCfg256x256, dt, LayoutB>(queue, p, num_wgs);
  } else if (tile == kTile128x128) {
    run_m_grouped_cfg<TileCfg128x128, dt, LayoutB>(queue, p, num_wgs);
  } else if (tile == kTile128x256) {
    run_m_grouped_cfg<TileCfg128x256, dt, LayoutB>(queue, p, num_wgs);
  } else {
    run_m_grouped_cfg<TileCfg256x128, dt, LayoutB>(queue, p, num_wgs);
  }
}

template <DType dt>
void run_k_grouped(sycl::queue& queue, const Params& p) {
  if (p.num_groups <= 0 || p.N <= 0 || p.M <= 0) {
    return;
  }
  const int tile = choose_tile(queue.get_device(), p);
  const int num_wgs = p.num_workgroups > 0
      ? p.num_workgroups
      : default_num_workgroups(queue.get_device(), tile);
  if (tile == kTile256x256) {
    run_k_grouped_cfg<TileCfg256x256, dt>(queue, p, num_wgs);
  } else if (tile == kTile128x128) {
    run_k_grouped_cfg<TileCfg128x128, dt>(queue, p, num_wgs);
  } else if (tile == kTile128x256) {
    run_k_grouped_cfg<TileCfg128x256, dt>(queue, p, num_wgs);
  } else {
    run_k_grouped_cfg<TileCfg256x128, dt>(queue, p, num_wgs);
  }
}

} // namespace grouped_gemm
} // namespace sycltla
