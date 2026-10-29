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
#include <type_traits>

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
// The production instantiations use the configs below.
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
  // One split barrier every two K steps instead of every step, and
  // sub-groups whose 32x64 accumulator block lies outside D skip the
  // mainloop (see BarrierEveryOf / SkipIdleSgOf).
  static constexpr int kBarrierEvery = 2;
  static constexpr bool kSkipIdleSg = true;
};

// grad_weight (K-grouped) variant of the 256x256 tile: K step 16. Each
// expert contributes only a short contraction (~190 rows at TP4/EP4), so a
// finer K step keeps more loads in flight through the short mainloop. Same
// output tile and sub-group layout as TileCfg256x256, so the tile id, the
// work-group size and the default grid are unchanged.
struct TileCfg256x256KGrouped {
  using WGTile = Shape<_256, _256, _16>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  static constexpr int kTileM = 256;
  static constexpr int kTileN = 256;
  static constexpr int kTileK = 16;
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

// Optional Cfg::kBarrierEvery: the work-group split barrier spans this many
// K tiles (default 1 = arrive/wait every K tile). Larger values let the
// sub-groups sharing an XVE drift apart by more than one K step, so one
// sub-group's DPAS work can cover another's load latency on long K loops.
template <class Cfg, class = void>
struct BarrierEveryOf {
  static constexpr int value = 1;
};
template <class Cfg>
struct BarrierEveryOf<Cfg, std::void_t<decltype(Cfg::kBarrierEvery)>> {
  static constexpr int value = Cfg::kBarrierEvery;
};

// Optional Cfg::kSkipIdleSg (default false): a sub-group whose whole
// accumulator slab lies outside D (ragged last row tile of a group, or a
// partial last N tile) skips its A/B block loads and DPAS work; it still
// joins the work-group barriers and the cooperative prefetches. With ~190
// rows per expert on 256-row tiles, 2 of 8 sub-group rows are idle, which
// frees XMX issue slots and L1 load bandwidth for the busy sub-groups that
// share their XVEs.
template <class Cfg, class = void>
struct SkipIdleSgOf {
  static constexpr bool value = false;
};
template <class Cfg>
struct SkipIdleSgOf<Cfg, std::void_t<decltype(Cfg::kSkipIdleSg)>> {
  static constexpr bool value = bool(Cfg::kSkipIdleSg);
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
    int BarrierEvery = 1,
    bool SkipIdle = false,
    class ATensor,
    class BTensor,
    class DTensor,
    class TiledMMA>
CUTE_DEVICE int gemm_tile(
    ATensor const& A,
    BTensor const& B,
    DTensor& D,
    int m_idx,
    int n_idx,
    TiledMMA const& mma,
    int pre_issued, // leading K tiles already prefetched by the previous tile
    ATensor const& An, // operands of this work-group's next tile (prefetch)
    BTensor const& Bn,
    int mn_idx,
    int nn_idx,
    bool has_next) {
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
  // Epilogue store: the auto selector picks block width = one DPAS N atom
  // (16 elements), i.e. 8x16 messages. A 2D store may be 64 bytes wide, so
  // use 8 x (512 / bits) blocks when each sub-group's N slab allows it:
  // half the store messages per tile (the reorder into the store fragment
  // happens anyway for the fp32 -> bf16 conversion).
  using DElem = typename DTensor::value_type;
  constexpr int kDBits = int(sizeof_bits_v<DElem>);
  constexpr int kStoreW = 512 / kDBits;
  using MmaTileN = std::decay_t<decltype(get<1>(
      std::declval<TiledMMA const&>().tile_mnk()))>;
  using MmaThrN = std::decay_t<decltype(size<2>(
      std::declval<TiledMMA const&>().get_thr_layout_vmnk()))>;
  constexpr int kSgN = int(MmaTileN::value) / int(MmaThrN::value);
  using StoreOp = std::conditional_t<
      (kSgN % kStoreW == 0),
      XE_STORE_2D<kDBits, 8, kStoreW>,
      void>;
  auto tiled_copy_d = get_block_2d_copy_D<StoreOp>(mma, D);

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

  // Prefetch view of the next tile: the prefetch pipeline continues across
  // tile boundaries, so the next tile's first K tiles are requested during
  // this tile's last mainloop iterations (not after its store).
  Tensor cAn = make_identity_tensor(An.shape());
  Tensor cBn = make_identity_tensor(Bn.shape());
  Tensor gAn = local_tile(cAn, select<0, 2>(wg_tile), make_coord(mn_idx, _));
  Tensor gBn = local_tile(cBn, select<1, 2>(wg_tile), make_coord(nn_idx, _));
  auto prefetch_an = make_block_2d_prefetch(get_block_2d_copy_A<void>(mma, An));
  auto prefetch_bn = make_block_2d_prefetch(get_block_2d_copy_B<void>(mma, Bn));
  auto pAgAn = prefetch_an.get_slice(local_id).partition_S(gAn);
  auto pBgBn = prefetch_bn.get_slice(local_id).partition_S(gBn);

  constexpr int barrier_scope = 2;
  constexpr int prefetch_dist = PrefetchDist;
  const int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int next_limit = 0;
  if (has_next) {
    const int next_k_count = ceil_div(shape<1>(An), get<2>(wg_tile));
    next_limit = next_k_count < prefetch_dist ? next_k_count : prefetch_dist;
  }
  int next_issued = 0;
  int prefetch_k = pre_issued < k_tile_count ? pre_issued : k_tile_count;

  clear(tCrD);

  // Does this sub-group own any in-bounds output element? Only the slab's
  // minimum corner matters: element 0 of each lane lies on the slab's first
  // row, and lane 0 holds its first column. Uniform over the sub-group.
  bool sg_active = true;
  if constexpr (SkipIdle) {
    Tensor tCcD = thr_mma.partition_C(gD);
    auto c0 = tCcD(0);
    const bool lane_active = (int(get<0>(c0)) < int(shape<0>(D))) &&
        (int(get<1>(c0)) < int(shape<1>(D)));
    sg_active = sycl::any_of_group(
        sycl::ext::oneapi::this_work_item::get_sub_group(), lane_active);
  }

  // Bounded by k_tile_count: with an empty contraction (K == 0, e.g. an
  // empty group in the K-grouped kernel) there is nothing to prefetch.
  CUTE_UNROLL
  for (; prefetch_k < prefetch_dist && prefetch_k < k_tile_count;
       prefetch_k++) {
    prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
    prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
  }

  static_assert(BarrierEvery >= 1, "kBarrierEvery must be >= 1");
  // Two copies of the mainloop: the busy one is exactly the unpredicated
  // production loop (no runtime branch around the copy fragments, so no
  // extra live ranges / spills); an idle sub-group runs only the barriers
  // and the cooperative prefetches, keeping the barrier count identical.
  auto mainloop = [&](auto active_tag) {
    constexpr bool kActive = decltype(active_tag)::value;
    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
      if constexpr (BarrierEvery == 1) {
        barrier_arrive(barrier_scope);
      } else if (k_tile % BarrierEvery == 0) {
        barrier_arrive(barrier_scope);
      }

      if constexpr (kActive) {
        copy(tiled_copy_a, tAgA(_, _, _, k_tile), tArA);
        copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBrB);
      }

      if (prefetch_k < k_tile_count) {
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
      } else if (next_issued < next_limit) {
        prefetch(prefetch_an, pAgAn(_, _, _, next_issued));
        prefetch(prefetch_bn, pBgBn(_, _, _, next_issued));
        ++next_issued;
      }

      if constexpr (kActive) {
        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        cute::gemm(mma, tCrA, tCrB, tCrD);
      }
      if constexpr (BarrierEvery == 1) {
        barrier_wait(barrier_scope);
      } else if (
          (k_tile % BarrierEvery) == BarrierEvery - 1 ||
          k_tile == k_tile_count - 1) {
        barrier_wait(barrier_scope);
      }
    }
  };
  if constexpr (SkipIdle) {
    if (sg_active) {
      mainloop(std::true_type{});
    } else {
      mainloop(std::false_type{});
      return next_issued; // slab fully out of bounds: the store is a no-op
    }
  } else {
    mainloop(std::true_type{});
  }

  reorder(tCrD, tCrD_final);
  copy(tiled_copy_d, tCrD_final, tCgD);
  return next_issued;
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

  // Sub-group-parallel variant of locate(): each lane reads the offs entries
  // of one group (16 groups per round trip), an inclusive scan over the
  // sub-group yields the cumulative tile counts, and the first lane whose
  // cumulative end exceeds `linear` owns the tile. The serial walk issues
  // one dependent offs load per group crossed (~10 L3 round trips before
  // every tile on small-group cases); this needs one per 16 groups.
  // Must be called in sub-group-uniform control flow with uniform `linear`.
  // With linear < 0 it only counts: returns false and sets `total`.
  CUTE_DEVICE bool locate_par(
      int64_t linear,
      GroupCursor& c,
      int32_t& m_idx,
      int32_t& n_idx,
      int64_t* total = nullptr) const {
    // Fast path: the next tile of a persistent work-group is often still in
    // the group it already holds (large groups): no offs loads, no
    // sub-group collectives. `c` is uniform across the sub-group.
    if (linear >= 0 && c.tiles > 0 && linear >= c.tile_start &&
        linear < c.tile_start + c.tiles) {
      const int32_t local = int32_t(linear - c.tile_start);
      m_idx = local % c.tiles_m;
      n_idx = local / c.tiles_m;
      return true;
    }
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    const int lane = int(sg.get_local_linear_id());
    const int nlanes = int(sg.get_local_linear_range());
    const int32_t tiles_n = (n + TileN - 1) / TileN;
    // Tile counts are scanned in 32-bit (native SIMD ops instead of
    // emulated 64-bit adds/shuffles in the scan and broadcasts).
    int32_t base = 0;
    for (int32_t g0 = 0; g0 < num_groups; g0 += nlanes) {
      const int32_t g = g0 + lane;
      const bool in_range = g < num_groups;
      int32_t start = 0, end = 0;
      if (in_range) {
        start = (g == 0) ? 0 : offs[g - 1];
        end = offs[g];
      }
      start = start < 0 ? 0 : start;
      const int32_t rows = (end > start) ? (end - start) : 0;
      const int32_t m = KGrouped ? fixed_m : rows;
      const int32_t tiles_m = (m + TileM - 1) / TileM;
      const int32_t tiles = in_range ? tiles_m * tiles_n : 0;
      const int32_t incl = base +
          sycl::inclusive_scan_over_group(sg, tiles, sycl::plus<int32_t>());
      const bool hit = in_range && (linear >= 0) && (int64_t(incl) > linear);
      const int first = sycl::reduce_over_group(
          sg, hit ? lane : nlanes, sycl::minimum<int>());
      if (first < nlanes) {
        c.g = g0 + first;
        c.row_start = sycl::select_from_group(sg, start, first);
        c.rows = sycl::select_from_group(sg, rows, first);
        c.tiles_m = sycl::select_from_group(sg, tiles_m, first);
        const int32_t t = sycl::select_from_group(sg, tiles, first);
        const int32_t te = sycl::select_from_group(sg, incl, first);
        c.tiles = int64_t(t);
        c.tile_start = int64_t(te - t);
        const int32_t local = int32_t(linear - c.tile_start);
        m_idx = local % c.tiles_m;
        n_idx = local / c.tiles_m;
        return true;
      }
      base = sycl::select_from_group(sg, incl, nlanes - 1);
    }
    if (total != nullptr) {
      *total = int64_t(base);
    }
    return false;
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

    const int64_t wg = item.get_group_linear_id();
    const int64_t nwg = int64_t(item.get_group_range(0)) *
        item.get_group_range(1) * item.get_group_range(2);

    T* a = const_cast<T*>(static_cast<const T*>(p.a));
    T* b = const_cast<T*>(static_cast<const T*>(p.b));
    T* d = static_cast<T*>(p.d);

    auto make_a = [&](const GroupCursor& c) {
      return make_2d_tensor<'R'>(
          a + int64_t(c.row_start) * p.lda, c.rows, p.K, int(p.lda));
    };
    auto make_b = [&](const GroupCursor& c) {
      return make_2d_tensor<LayoutB>(
          b + int64_t(c.g) * p.group_stride_b, p.N, p.K, int(p.ldb));
    };

    // Device-side wave-tail balancing (no scratch, only offs): count all
    // tiles; if the last partial wave holds r tiles and S*r <= nwg, those r
    // tiles are run as S*r narrower (TileN/S) sub-tiles, one per work-group,
    // so the final wave takes ~1/S of a full tile instead of a full tile
    // with most of the GPU idle. S = 4 when it fits, else 2.
    int64_t total = 0;
    {
      GroupCursor c;
      int32_t tm = 0, tn = 0;
      sched.locate_par(-1, c, tm, tn, &total);
    }
    const int64_t rem = total % nwg;
    int split = 1;
    if (rem > 0) {
      if (kSplit4Ok && 4 * rem <= nwg) {
        split = 4;
      } else if (kSplit2Ok && 2 * rem <= nwg) {
        split = 2;
      }
    }
    const int64_t full_end = (split > 1) ? (total - rem) : total;

    int32_t m_idx = 0, n_idx = 0;
    bool valid =
        (int64_t(wg) < full_end) && sched.locate_par(wg, cur, m_idx, n_idx);
    int pre_issued = 0;
    for (int64_t lin = wg; valid; lin += nwg) {
      GroupCursor nxt = cur;
      int32_t mn_idx = 0, nn_idx = 0;
      const bool has_next = (lin + nwg < full_end) &&
          sched.locate_par(lin + nwg, nxt, mn_idx, nn_idx);
      const GroupCursor& pc = has_next ? nxt : cur;
      auto D = make_2d_tensor<'R'>(
          d + int64_t(cur.row_start) * p.ldd, cur.rows, p.N, int(p.ldd));
      pre_issued = gemm_tile<
          Cfg::kPrefetchDist,
          BarrierEveryOf<Cfg>::value,
          SkipIdleSgOf<Cfg>::value>(
          make_a(cur),
          make_b(cur),
          D,
          m_idx,
          n_idx,
          mma,
          pre_issued,
          make_a(pc),
          make_b(pc),
          mn_idx,
          nn_idx,
          has_next);
      cur = nxt;
      m_idx = mn_idx;
      n_idx = nn_idx;
      valid = has_next;
    }

    if (split > 1 && int64_t(wg) < int64_t(split) * rem) {
      GroupCursor c;
      int32_t tm = 0, tn = 0;
      const int64_t lin = full_end + int64_t(wg) / split;
      const int q = int(int64_t(wg) % split);
      if (sched.locate_par(lin, c, tm, tn)) {
        auto A = make_a(c);
        auto B = make_b(c);
        auto D = make_2d_tensor<'R'>(
            d + int64_t(c.row_start) * p.ldd, c.rows, p.N, int(p.ldd));
        if constexpr (kSplit4Ok) {
          if (split == 4) {
            run_sub<4>(A, B, D, tm, tn * 4 + q);
            return;
          }
        }
        if constexpr (kSplit2Ok) {
          run_sub<2>(A, B, D, tm, tn * 2 + q);
        }
      }
    }
  }

  static constexpr int kThrN =
      int(decltype(size<1>(typename Cfg::SGLayout{}))::value);
  // A sub-tile needs >= one 16-wide DPAS N atom per sub-group.
  static constexpr bool kSplit2Ok = (Cfg::kTileN % (2 * 16 * kThrN)) == 0;
  static constexpr bool kSplit4Ok = (Cfg::kTileN % (4 * 16 * kThrN)) == 0;

  template <int S>
  using SubMma = typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, T, T>>,
      Layout<Shape<Int<Cfg::kTileM>, Int<Cfg::kTileN / S>, Int<Cfg::kTileK>>>,
      typename Cfg::SGLayout>::TiledMMA;

  template <int S, class AT, class BT, class DT>
  CUTE_DEVICE void run_sub(
      AT const& A,
      BT const& B,
      DT& D,
      int m_idx,
      int sub_n) const {
    if (int64_t(sub_n) * (Cfg::kTileN / S) >= int64_t(p.N)) {
      return; // sub-tile fully past N (ragged N): nothing to compute/store
    }
    const auto smma = SubMma<S>{};
    gemm_tile<
        Cfg::kPrefetchDist,
        BarrierEveryOf<Cfg>::value,
        SkipIdleSgOf<Cfg>::value>(
        A, B, D, m_idx, sub_n, smma, 0, A, B, m_idx, sub_n, false);
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

    const int64_t wg = item.get_group_linear_id();
    const int64_t nwg = int64_t(item.get_group_range(0)) *
        item.get_group_range(1) * item.get_group_range(2);

    T* a = const_cast<T*>(static_cast<const T*>(p.a));
    T* b = const_cast<T*>(static_cast<const T*>(p.b));
    T* d = static_cast<T*>(p.d);

    // A: (M, rows) with element (m, r) at m + r * lda  -> column-major.
    auto make_a = [&](const GroupCursor& c) {
      return make_2d_tensor<'C'>(
          a + int64_t(c.row_start) * p.lda, p.M, c.rows, int(p.lda));
    };
    // B as (N, rows) with element (n, r) at n + r * ldb -> column-major
    // view of the row-major [rows, N] operand.
    auto make_b = [&](const GroupCursor& c) {
      return make_2d_tensor<'C'>(
          b + int64_t(c.row_start) * p.ldb, p.N, c.rows, int(p.ldb));
    };

    int32_t m_idx = 0, n_idx = 0;
    bool valid = sched.locate_par(wg, cur, m_idx, n_idx);
    int pre_issued = 0;
    for (int64_t lin = wg; valid; lin += nwg) {
      GroupCursor nxt = cur;
      int32_t mn_idx = 0, nn_idx = 0;
      const bool has_next = sched.locate_par(lin + nwg, nxt, mn_idx, nn_idx);
      const GroupCursor& pc = has_next ? nxt : cur;
      auto D = make_2d_tensor<'R'>(
          d + int64_t(cur.g) * p.group_stride_d, p.M, p.N, int(p.ldd));
      pre_issued = gemm_tile<
          Cfg::kPrefetchDist,
          BarrierEveryOf<Cfg>::value,
          SkipIdleSgOf<Cfg>::value>(
          make_a(cur),
          make_b(cur),
          D,
          m_idx,
          n_idx,
          mma,
          pre_issued,
          make_a(pc),
          make_b(pc),
          mn_idx,
          nn_idx,
          has_next);
      cur = nxt;
      m_idx = mn_idx;
      n_idx = nn_idx;
      valid = has_next;
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
    run_k_grouped_cfg<TileCfg256x256KGrouped, dt>(queue, p, num_wgs);
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
