/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/transformers/xpu/flash_attn/sycltla/mha_bwd.h>
#include <ATen/native/transformers/xpu/flash_attn/sycltla/mha_common.h>

// batch, numhead_qo,numhead_kv,seqlen_qo,seqlen_kv,headsize_qk,headsize_vo
using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>;

namespace cute {

template <typename Layout>
auto convert_layout_2d_layout(Layout layout) {
  auto l =
      make_layout(make_layout(get<0>(layout), get<1>(layout)), get<2>(layout));
  return l;
}

template <bool Is_even_M, class T>
void compute_o_dot_do(
    T& trait,
    Param<typename T::DType>& param,
    const int m_block,
    const int bidb,
    const int bidh) {
  // The thread index.
  constexpr int kBlockM = T::kBlockM;
  constexpr int kHeadDim = T::kHeadDim;
  constexpr int kNSGs = T::kNSGs;
  constexpr int SubgroupSize = T::SubgroupSize;
  using DType = typename T::DType;
  using VType = typename T::VType;

  auto sg = compat::get_nd_item<1>().get_sub_group();
  auto bofst = Boffset(param);

  const index_t o_offset = bofst.o_offset(bidb, bidh, m_block * kBlockM);
  const index_t do_offset = bofst.do_offset(bidb, bidh, m_block * kBlockM);
  const index_t dqaccum_offset =
      bofst.dqaccum_offset(bidb, bidh, m_block * kBlockM);
  const index_t dpsum_offset = bofst.lse_offset(bidb, bidh, m_block * kBlockM);

  using ShapeO =
      Shape<std::conditional_t<Is_even_M, Int<kBlockM>, int>, Int<kHeadDim>>;
  using ShapeP = Shape<std::conditional_t<Is_even_M, Int<kBlockM>, int>>;
  ShapeO O_shape;
  ShapeP dP_shape;
  if constexpr (Is_even_M) {
    O_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});
    dP_shape = make_shape(Int<kBlockM>{});
  } else {
    O_shape = make_shape(param.tail_m, Int<kHeadDim>{});
    dP_shape = make_shape(param.tail_m);
  }
  auto dQ_shape = make_shape(Int<kBlockM>{}, Int<kHeadDim>{});

  Tensor mdO = make_tensor(
      make_gmem_ptr(param.do_ptr + do_offset),
      make_layout(O_shape, make_stride(param.do_r_stride, _1{})));
  Tensor mO = make_tensor(
      make_gmem_ptr(param.o_ptr + o_offset),
      make_layout(O_shape, make_stride(param.o_r_stride, _1{})));
  Tensor mdQaccum = make_tensor(
      make_gmem_ptr(param.dqaccum_ptr + dqaccum_offset),
      make_layout(
          make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
          make_stride(param.head_dim, _1{})));
  Tensor mdPsum = make_tensor(
      make_gmem_ptr(param.odo_ptr + dpsum_offset),
      make_layout(dP_shape, Stride<_1>{}));

  auto tileload_odo = make_tiled_copy(
      Copy_Atom<UniversalCopy<DType>, DType>{},
      Layout<
          Shape<Int<kNSGs>, Int<SubgroupSize>>,
          Stride<Int<SubgroupSize>, _1>>{},
      Layout<Shape<_1, _1>>{});
  auto tileload_dq = make_tiled_copy(
      Copy_Atom<UniversalCopy<VType>, VType>{},
      Layout<Shape<Int<kNSGs>, Int<SubgroupSize>>>{},
      Layout<Shape<_1, _1>>{});
  auto thr_load_odo = tileload_odo.get_thread_slice(ThreadIdxX());
  auto thr_load_dq = tileload_dq.get_thread_slice(ThreadIdxX());

  Tensor thr_tile_do_S = thr_load_odo.partition_S(mdO);
  Tensor thr_tile_o_S = thr_load_odo.partition_S(mO);
  Tensor thr_tile_dq_D = thr_load_dq.partition_D(mdQaccum);
  Tensor rdQ = make_fragment_like(thr_tile_dq_D);
  Tensor rdO = make_fragment_like<DType>(rdQ);
  Tensor rO = make_fragment_like<DType>(rdQ);
  clear(rdQ);
  copy(tileload_dq, rdQ, thr_tile_dq_D);

  Tensor cO = make_identity_tensor(dQ_shape);
  Tensor tcO = thr_load_odo.partition_S(cO);
  Tensor tcO_row = logical_divide(tcO, Shape<_1>{})(make_coord(0, 0), _, 0);
  Tensor rdO_2d =
      make_tensor(rdO.data(), convert_layout_2d_layout(rdO.layout()));
  Tensor rO_2d = make_tensor(rO.data(), convert_layout_2d_layout(rO.layout()));
  if constexpr (Is_even_M) {
    for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
      copy(tileload_odo, thr_tile_do_S(_, mi, _), rdO(_, mi, _));
      copy(tileload_odo, thr_tile_o_S(_, mi, _), rO(_, mi, _));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
      float accum = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1>(rdO_2d); ++ni) {
        accum = accum + (float)rdO_2d(mi, ni) * (float)rO_2d(mi, ni);
      }
      accum = sycl::reduce_over_group(sg, accum, sycl::plus<>());
      if (sg.get_local_id() == 0) {
        mdPsum(get<0>(tcO_row(mi))) = accum;
      }
    }
  } else {
    for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
      if (get<0>(tcO_row(mi)) < param.tail_m) {
        copy(tileload_odo, thr_tile_do_S(_, mi, _), rdO(_, mi, _));
        copy(tileload_odo, thr_tile_o_S(_, mi, _), rO(_, mi, _));
      }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size<0>(rdO_2d); ++mi) {
      float accum = 0.0f;
      CUTLASS_PRAGMA_UNROLL
      for (int ni = 0; ni < size<1>(rdO_2d); ++ni) {
        accum = accum + (float)rdO_2d(mi, ni) * (float)rO_2d(mi, ni);
      }
      accum = sycl::reduce_over_group(sg, accum, sycl::plus<>());
      if (sg.get_local_id() == 0 and get<0>(tcO_row(mi)) < param.tail_m)
        mdPsum(get<0>(tcO_row(mi))) = accum;
    }
  }
}

template <class T>
void mha_dot_do_o(T trait, Param<typename T::DType> param) {
  // The block index for the M dimension.
  const int m_block = BlockIdxX();
  // The block index for the batch.
  const int bidb = BlockIdxZ();
  // The block index for the head.
  const int bidh = BlockIdxY();
  ;
  if (m_block == param.m_block - 1 and param.tail_m > 0) {
    compute_o_dot_do<false>(trait, param, m_block, bidb, bidh);
  } else {
    compute_o_dot_do<true>(trait, param, m_block, bidb, bidh);
  }
}

template <
    typename Tensor0,
    typename Tensor1,
    typename Tensor2,
    typename Tensor3,
    typename Tensor4,
    typename Tensor5,
    typename Tensor6,
    typename Tensor7,
    typename Tensor8,
    typename TiledMma,
    typename TileMNK,
    typename TiledCopyA,
    typename TiledCopyB>
CUTLASS_DEVICE void gemm_ker(
    Tensor0& tCrCmn,
    Tensor1& tCrA,
    Tensor2& tCrB,
    Tensor3& tAgAmk,
    Tensor4& tArA,
    Tensor5& gA,
    Tensor6& tBgBnk,
    Tensor7& tBrB,
    Tensor8& gB,
    TiledMma& tiled_mma,
    TileMNK& tile_mnk,
    TiledCopyA& copy_a,
    TiledCopyB& copy_b) {
  constexpr int barrier_scope = 2;
  CUTLASS_PRAGMA_UNROLL
  for (int m = 0; m < size<3>(tAgAmk); ++m) {
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<3>(tBgBnk); ++n) {
      auto tCrC = tCrCmn(_, _, _, m, n);
      auto tAgA = tAgAmk(_, _, _, m, _);
      auto tBgB = tBgBnk(_, _, _, n, _);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<3>(tAgA); ++k) {
        barrier_arrive(barrier_scope);
        cute::copy(copy_a, tAgA(_, _, _, k), tArA);
        cute::copy(copy_b, tBgB(_, _, _, k), tBrB);
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);
      }
    }
  }
}

template <
    typename Engine0,
    typename Layout0,
    typename Engine1,
    typename Layout1>
CUTLASS_DEVICE void apply_mask_causal(
    Tensor<Engine0, Layout0>& tensor,
    Tensor<Engine1, Layout1>& rC,
    int m_offset,
    int n_offset,
    int diagonal_offset = 0) {
  auto sg = compat::get_nd_item<1>().get_sub_group();
  int sg_local_id = sg.get_local_id();
  Tensor rC_2d = make_tensor(rC.data(), convert_layout_2d_layout(rC.layout()));
  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < size<1>(tensor); ++n) {
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tensor); ++m) {
      int x = n_offset + get<1>(rC_2d(m, n)) + sg_local_id;
      int y = m_offset + get<0>(rC_2d(m, n)) + diagonal_offset;
      if (x > y) {
        tensor(m, n) = -INFINITY;
      }
    }
  }
  return;
}

template <
    bool Is_even_MN,
    class TileCopy,
    class Engine0,
    class Layout0,
    class Engine1,
    class Layout1>
CUTLASS_DEVICE void mha_save(
    TileCopy& tile_copy,
    Tensor<Engine0, Layout0>& src,
    Tensor<Engine1, Layout1>& dst) {
  static_assert(Layout0::rank == 5, "Only support Tensor with 5 ranks");
  static_assert(
      Layout0::rank == Layout1::rank, "Only support same rank Tensor");
  if constexpr (Is_even_MN) {
    copy(tile_copy, src, dst);
  } else {
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<3>(dst); ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < size<4>(dst); ++n) {
        auto src_block = src(_, _, _, m, n);
        auto dst_block = dst(_, _, _, m, n);
        copy(tile_copy, src_block, dst_block);
      }
    }
  }
}

template <
    bool Is_even_MN,
    class TileCopy,
    class Engine0,
    class Layout0,
    class Engine1,
    class Layout1>
CUTLASS_DEVICE void mha_load(
    TileCopy& tile_copy,
    Tensor<Engine0, Layout0>& src,
    Tensor<Engine1, Layout1>& dst) {
  static_assert(Layout0::rank == 5, "Only support Tensor with 5 ranks");
  static_assert(
      Layout0::rank == Layout1::rank, "Only support same rank Tensor");
  if constexpr (Is_even_MN) {
    copy(tile_copy, src, dst);
  } else {
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<3>(src); ++m) {
      auto src_block = src(_, _, _, m, _);
      auto dst_block = dst(_, _, _, m, _);
      copy(tile_copy, src_block, dst_block);
    }
  }
}

template <
    class Engine0,
    class Layout0,
    class Engine1,
    class Layout1,
    class Engine2,
    class Layout2>
CUTLASS_DEVICE void mha_atomic_add(
    Tensor<Engine0, Layout0>& m_tile,
    Tensor<Engine1, Layout1>& g_tile,
    Tensor<Engine2, Layout2>& r_tile,
    const int local_id) {
  CUTLASS_PRAGMA_UNROLL
  for (int mi = 0; mi < size<3>(g_tile); ++mi) {
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<4>(g_tile); ++ni) {
      auto g = g_tile(_, _, _, mi, ni);
      auto r = r_tile(_, _, _, mi, ni);
      CUTLASS_PRAGMA_UNROLL
      for (int ki = 0; ki < size(g); ++ki) {
        auto [m, n, l] = g(ki);
        cutlass::atomicAdd(&m_tile(m, n + local_id, 0), r(ki));
      }
    }
  }
}

template <bool Is_even_M, class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void load_1colvec(
    Tensor0& reg,
    Tensor1& mT,
    Tensor2& coord_row,
    int tail_m = 0) {
  if constexpr (Is_even_M) {
    CUTLASS_PRAGMA_UNROLL
    for (int mi = 0; mi < size(reg); ++mi) {
      reg(mi) = mT(get<0>(coord_row(mi)));
    }
  } else {
    for (int mi = 0; mi < size(reg); ++mi) {
      int row = get<0>(coord_row(mi));
      if (row < tail_m) {
        reg(mi) = mT(row);
      }
    }
  }
}

template <typename Layout>
CUTLASS_DEVICE auto convert_layout_acc_layout(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 8);
  static_assert(decltype(rank(acc_layout))::value == 5);
  auto l = logical_divide(acc_layout, Shape<_1>{}); // ((2, 2), MMA_M, MMA_N)
  auto l2 = make_layout(
      make_layout(get<0, 1>(l), get<1>(l), get<3>(l)),
      make_layout(get<0, 0>(l), get<4>(l)));
  return l2;
}

template <class Engine0, class Layout0, class Engine1, class Layout1>
CUTLASS_DEVICE void scale_apply_exp2(
    Tensor<Engine0, Layout0>& tensor,
    Tensor<Engine1, Layout1>& max,
    const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
  CUTLASS_PRAGMA_UNROLL
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * M_LOG2E;
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

template <class Tensor0, class Tensor1, class Tensor2>
CUTLASS_DEVICE void softmax_backward(
    Tensor0& P,
    Tensor1& dP_sum,
    Tensor2& dP,
    const float scale) {
  CUTLASS_PRAGMA_UNROLL
  for (int mi = 0; mi < size<0>(dP); ++mi) {
    CUTLASS_PRAGMA_UNROLL
    for (int mj = 0; mj < size<1>(dP); ++mj) {
      dP(mi, mj) = P(mi, mj) * (dP(mi, mj) - dP_sum(mi)) * scale;
    }
  }
}

template <class CVT, class T0, class T1>
CUTLASS_DEVICE auto convert_type(CVT& cvt, T0& src, T1& dst) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(src); ++i) {
    dst(i) = cvt(src(i));
  }
  return dst;
}

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}
template <
    typename T,
    class Tensor0,
    class Tensor1,
    class Tensor2,
    class ThrMMA,
    class TiledMma,
    class TileMNK,
    class TileloadA,
    class TileloadB>
CUTLASS_DEVICE void gemm_S(
    Tensor0& tSrS,
    Tensor1& gQ,
    Tensor2& gKtV,
    ThrMMA& thr_mma_sdp,
    TiledMma& tiled_mma_sdp,
    TileMNK& tile_sdp,
    TileloadA& tileloadQ,
    TileloadB& tileloadKt) {
  Tensor tSgQ = thr_mma_sdp.partition_A(gQ);
  Tensor tSgKt = thr_mma_sdp.partition_B(gKtV);

  Tensor tSrQ = make_tensor<T>(
      make_fragment_layout(tileloadQ, tSgQ(_, _, _, 0, 0).shape()));
  Tensor tSrKt = make_tensor<T>(
      make_fragment_layout(tileloadKt, tSgKt(_, _, _, 0, 0).shape()));

  ThrCopy thr_copy_q = tileloadQ.get_slice(compat::local_id::x());
  ThrCopy thr_copy_kt = tileloadKt.get_slice(compat::local_id::x());

  Tensor tQrQ = thr_copy_q.retile_D(tSrQ);
  Tensor tKtrKt = thr_copy_kt.retile_D(tSrKt);

  Tensor tQgQ = thr_copy_q.retile_S(tSgQ);
  Tensor tKtgKt = thr_copy_kt.retile_S(tSgKt);

  gemm_ker(
      tSrS,
      tSrQ,
      tSrKt,
      tQgQ,
      tQrQ,
      gQ,
      tKtgKt,
      tKtrKt,
      gKtV,
      tiled_mma_sdp,
      tile_sdp,
      tileloadQ,
      tileloadKt);
}

template <
    typename T,
    class Tensor0,
    class Tensor1,
    class Tensor2,
    class ThrMma,
    class TiledMma,
    class TileMNK,
    class TileloadA,
    class TileloadB>
CUTLASS_DEVICE void gemm_dP(
    Tensor0& tdPrdP,
    Tensor1& gdO,
    Tensor2& gKtV,
    ThrMma& thr_mma_sdp,
    TiledMma& tiled_mma_sdp,
    TileMNK& tile_sdp,
    TileloadA& tileloaddO,
    TileloadB& tileloadV) {
  Tensor tdPgdO = thr_mma_sdp.partition_A(gdO);
  Tensor tdPgV = thr_mma_sdp.partition_B(gKtV);

  Tensor tdPrdO = make_tensor<T>(
      make_fragment_layout(tileloaddO, tdPgdO(_, _, _, 0, 0).shape()));
  Tensor tdPrV = make_tensor<T>(
      make_fragment_layout(tileloadV, tdPgV(_, _, _, 0, 0).shape()));

  ThrCopy thr_copy_do = tileloaddO.get_slice(compat::local_id::x());
  ThrCopy thr_copy_v = tileloadV.get_slice(compat::local_id::x());

  Tensor tdOrdO = thr_copy_do.retile_D(tdPrdO);
  Tensor tVrV = thr_copy_v.retile_D(tdPrV);

  Tensor tdOgdO = thr_copy_do.retile_S(tdPgdO);
  Tensor tVgV = thr_copy_v.retile_S(tdPgV);

  gemm_ker(
      tdPrdP,
      tdPrdO,
      tdPrV,
      tdOgdO,
      tdOrdO,
      gdO,
      tVgV,
      tVrV,
      gKtV,
      tiled_mma_sdp,
      tile_sdp,
      tileloaddO,
      tileloadV);
}

template <
    typename T,
    class Tensor0,
    class Tensor1,
    class Tensor2,
    class ThrMma,
    class TiledMma,
    class TileMNK,
    class TileloadA,
    class TileloadB>
CUTLASS_DEVICE void gemm_dV(
    Tensor0& tdVrdV,
    Tensor1& gPt,
    Tensor2& gQtdOt,
    ThrMma& thr_mma_dkv,
    TiledMma& tiled_mma_dkv,
    TileMNK& tile_dkv,
    TileloadA& tileloadPt,
    TileloadB& tileloaddOt) {
  Tensor tdVgPt = thr_mma_dkv.partition_A(gPt);
  Tensor tdVgdOt = thr_mma_dkv.partition_B(gQtdOt);

  Tensor tdVrPt = make_tensor<T>(
      make_fragment_layout(tileloadPt, tdVgPt(_, _, _, 0, 0).shape()));
  Tensor tdVrdOt = make_tensor<T>(
      make_fragment_layout(tileloaddOt, tdVgdOt(_, _, _, 0, 0).shape()));

  ThrCopy thr_copy_pt = tileloadPt.get_slice(compat::local_id::x());
  ThrCopy thr_copy_dot = tileloaddOt.get_slice(compat::local_id::x());

  Tensor tPtrPt = thr_copy_pt.retile_D(tdVrPt);
  Tensor tdOtrdOt = thr_copy_dot.retile_D(tdVrdOt);

  Tensor tPtgPt = thr_copy_pt.retile_S(tdVgPt);
  Tensor tdOtgdOt = thr_copy_dot.retile_S(tdVgdOt);

  gemm_ker(
      tdVrdV,
      tdVrPt,
      tdVrdOt,
      tPtgPt,
      tPtrPt,
      gPt,
      tdOtgdOt,
      tdOtrdOt,
      gQtdOt,
      tiled_mma_dkv,
      tile_dkv,
      tileloadPt,
      tileloaddOt);
}

template <
    typename T,
    class Tensor0,
    class Tensor1,
    class Tensor2,
    class ThrMma,
    class TiledMma,
    class TileMNK,
    class TileloadA,
    class TileloadB>
CUTLASS_DEVICE void gemm_dQ(
    Tensor0& tdQrdQ,
    Tensor1& gdPa,
    Tensor2& gK,
    ThrMma& thr_mma_dq,
    TiledMma& tiled_mma_dq,
    TileMNK& tile_dq,
    TileloadA& tileloaddP,
    TileloadB& tileloadK) {
  Tensor tdQgdP = thr_mma_dq.partition_A(gdPa);
  Tensor tdQgK = thr_mma_dq.partition_B(gK);

  Tensor tdQrdP = make_tensor<T>(
      make_fragment_layout(tileloaddP, tdQgdP(_, _, _, 0, 0).shape()));
  Tensor tdQrK = make_tensor<T>(
      make_fragment_layout(tileloadK, tdQgK(_, _, _, 0, 0).shape()));

  ThrCopy thr_copy_dp = tileloaddP.get_slice(compat::local_id::x());
  ThrCopy thr_copy_k = tileloadK.get_slice(compat::local_id::x());

  Tensor tdPrdPa = thr_copy_dp.retile_D(tdQrdP);
  Tensor tKrK = thr_copy_k.retile_D(tdQrK);

  Tensor tdPgdPa = thr_copy_dp.retile_S(tdQgdP);
  Tensor tKgK = thr_copy_k.retile_S(tdQgK);

  gemm_ker(
      tdQrdQ,
      tdQrdP,
      tdQrK,
      tdPgdPa,
      tdPrdPa,
      gdPa,
      tKgK,
      tKrK,
      gK,
      tiled_mma_dq,
      tile_dq,
      tileloaddP,
      tileloadK);
}

template <
    typename T,
    class Tensor0,
    class Tensor1,
    class Tensor2,
    class ThrMma,
    class TiledMma,
    class TileMNK,
    class TileloadA,
    class TileloadB>
CUTLASS_DEVICE void gemm_dK(
    Tensor0& tdKrdK,
    Tensor1& gdPt,
    Tensor2& gQtdOt,
    ThrMma& thr_mma_dkv,
    TiledMma& tiled_mma_dkv,
    TileMNK& tile_dkv,
    TileloadA& tileloaddPt,
    TileloadB& tileloadQt) {
  Tensor tdKgdPt = thr_mma_dkv.partition_A(gdPt);
  Tensor tdKgQt = thr_mma_dkv.partition_B(gQtdOt);

  Tensor tdKrdPt = make_tensor<T>(
      make_fragment_layout(tileloaddPt, tdKgdPt(_, _, _, 0, 0).shape()));
  Tensor tdKrQt = make_tensor<T>(
      make_fragment_layout(tileloadQt, tdKgQt(_, _, _, 0, 0).shape()));

  ThrCopy thr_copy_dpt = tileloaddPt.get_slice(compat::local_id::x());
  ThrCopy thr_copy_qt = tileloadQt.get_slice(compat::local_id::x());

  Tensor tdPtrdPt = thr_copy_dpt.retile_D(tdKrdPt);
  Tensor tQtrQt = thr_copy_qt.retile_D(tdKrQt);

  Tensor tdPtgdPt = thr_copy_dpt.retile_S(tdKgdPt);
  Tensor tQtgQt = thr_copy_qt.retile_S(tdKgQt);

  gemm_ker(
      tdKrdK,
      tdKrdPt,
      tdKrQt,
      tdPtgdPt,
      tdPtrdPt,
      gdPt,
      tQtgQt,
      tQtrQt,
      gQtdOt,
      tiled_mma_dkv,
      tile_dkv,
      tileloaddPt,
      tileloadQt);
}

template <
    bool is_causal,
    bool Is_even_N,
    typename T,
    typename V,
    int kBlockM,
    int kBlockN,
    class Tensor0,
    class Tensor1,
    class Tensor2,
    class Tensor3,
    class Tensor4,
    class Tensor5,
    class TilesaveP,
    class Param>
CUTLASS_DEVICE auto softmax_forward(
    Tensor0& scores,
    Tensor1& tSrS,
    Tensor2& taccScS,
    Tensor3& mLSE,
    Tensor4& mdPsum,
    Tensor5& tPgP,
    TilesaveP& tilesaveP,
    Param& param,
    int m_block,
    int n_block,
    bool Is_even_M,
    int tail_m = 0) {
  if constexpr (is_causal) {
    Tensor taccScS_rc = logical_divide(taccScS, Shape<_1>{});
    apply_mask_causal(
        scores,
        taccScS_rc,
        m_block * kBlockM,
        n_block * kBlockN,
        param.seq_len_kv - param.seq_len_q);
  }
  Tensor taccScS_row =
      logical_divide(taccScS, Shape<_1>{})(make_coord(0, _), _, 0);
  Tensor lse = make_tensor<V>(Shape<Int<decltype(size(taccScS_row))::value>>{});

  if (Is_even_M) {
    load_1colvec<true>(lse, mLSE, taccScS_row);
  } else {
    load_1colvec<false>(lse, mLSE, taccScS_row, tail_m);
  }

  // P=softmax(S,lse)
  scale_apply_exp2(scores, lse, param.scale_softmax_log2);
  if (Is_even_M)
    load_1colvec<true>(lse, mdPsum, taccScS_row);
  else
    load_1colvec<false>(lse, mdPsum, taccScS_row, tail_m);
  return lse;
}

template <bool Is_even_N, bool Seq_parallel, class Trait>
void dq_dk_dv_1colblock(
    Trait& trait,
    Param<typename Trait::DType>& param,
    const int bidb,
    const int bidh,
    const int bidhkv,
    const int n_block,
    const int tail_n = 0) {
  using T = typename Trait::DType;
  using V = typename Trait::VType;
  constexpr int kHeadDim = Trait::kHeadDim;
  constexpr int kBlockM = Trait::kBlockM;
  constexpr int kBlockN = Trait::kBlockN;
  constexpr bool is_causal = Trait::is_causal;
  auto sg = compat::get_nd_item<1>().get_sub_group();
  const int local_id = sg.get_local_id();
  auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;
  auto bofst = Boffset(param);

  const index_t q_offset = bofst.q_offset(bidb, bidh, 0);
  const index_t k_offset = bofst.k_offset(bidb, bidhkv, n_block * kBlockN);
  const index_t v_offset = bofst.v_offset(bidb, bidhkv, n_block * kBlockN);
  const index_t dk_offset = bofst.dk_offset(bidb, bidh, n_block * kBlockN);
  const index_t dv_offset = bofst.dv_offset(bidb, bidh, n_block * kBlockN);
  const index_t do_offset = bofst.do_offset(bidb, bidh, 0);
  const index_t dqaccum_offset = bofst.dqaccum_offset(bidb, bidh, 0);
  const index_t lse_offset = bofst.lse_offset(bidb, bidh, 0);
  // buff offset
  const index_t pb_offset =
      (bidb * param.num_head_q * param.seq_len_kv_pad * kBlockM +
       bidh * param.seq_len_kv_pad * kBlockM + n_block * kBlockN * kBlockM) *
      2;
  const index_t dsb_offset = pb_offset + kBlockN * kBlockM;

  const auto block_n_dim = tail_n == 0 ? Int<kBlockN>{} : ((tail_n + 1) & ~1);
  using Shape1 = Shape<
      std::conditional_t<Is_even_N, Int<kBlockN>, int>,
      Int<kHeadDim>,
      Int<1>>;
  using Shape2 = Shape<
      Int<kHeadDim>,
      std::conditional_t<Is_even_N, Int<kBlockN>, int>,
      Int<1>>;
  auto shapeQ = make_shape(kBlockM, Int<kHeadDim>{}, _1{});
  auto shapedQ = Shape<Int<kBlockM>, Int<kHeadDim>, _1>{};
  Shape1 shapeKtV;
  Shape2 shapeK;
  if constexpr (Is_even_N) {
    shapeKtV = make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, _1{});
    shapeK = make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, _1{});
  } else {
    shapeKtV = make_shape(tail_n, Int<kHeadDim>{}, _1{});
    shapeK = make_shape(Int<kHeadDim>{}, tail_n, _1{});
  }
  auto shapeO = make_shape(kBlockM, Int<kHeadDim>{}, _1{});
  auto shapeQtOt = make_shape(Int<kHeadDim>{}, kBlockM, _1{});

  auto shapeSP = make_shape(kBlockM, block_n_dim, _1{});

  auto shapePt = make_shape(block_n_dim, kBlockM, _1{});

  Tensor mQ = make_tensor(
      make_gmem_ptr(param.q_ptr + q_offset),
      make_layout(shapeQ, make_stride(param.q_r_stride, _1{}, _1{})));
  Tensor mKt = make_tensor(
      make_gmem_ptr(param.k_ptr + k_offset),
      make_layout(shapeKtV, make_stride(param.k_r_stride, _1{}, _1{})));
  Tensor mV = make_tensor(
      make_gmem_ptr(param.v_ptr + v_offset),
      make_layout(shapeKtV, make_stride(param.v_r_stride, _1{}, _1{})));
  Tensor mdO = make_tensor(
      make_gmem_ptr(param.do_ptr + do_offset),
      make_layout(shapeO, make_stride(param.do_r_stride, _1{}, _1{})));
  // intermediate buffer
  Tensor mP = make_tensor(
      make_gmem_ptr(param.pb_ptr + pb_offset),
      make_layout(shapeSP, make_stride(block_n_dim, _1{}, _1{})));
  Tensor mPt = make_tensor(
      make_gmem_ptr(param.pb_ptr + pb_offset),
      make_layout(shapePt, make_stride(_1{}, block_n_dim, _1{})));
  Tensor mdOt = make_tensor(
      make_gmem_ptr(param.do_ptr + do_offset),
      make_layout(shapeQtOt, make_stride(_1{}, param.do_r_stride, _1{})));
  Tensor mK = make_tensor(
      make_gmem_ptr(param.k_ptr + k_offset),
      make_layout(shapeK, make_stride(_1{}, param.k_r_stride, _1{})));
  Tensor mdPt = make_tensor(
      make_gmem_ptr(param.pb_ptr + dsb_offset),
      make_layout(shapePt, make_stride(_1{}, block_n_dim, _1{})));
  Tensor mQt = make_tensor(
      make_gmem_ptr(param.q_ptr + q_offset),
      make_layout(shapeQtOt, make_stride(_1{}, param.q_r_stride, _1{})));

  Tensor mLSE = make_tensor(
      make_gmem_ptr(param.lse_ptr + lse_offset),
      make_layout(Shape<Int<kBlockM>>{}, Stride<_1>{}));
  Tensor mdPsum = make_tensor(
      make_gmem_ptr(param.odo_ptr + lse_offset),
      make_layout(Shape<Int<kBlockM>>{}, Stride<_1>{}));

  Tensor mdV = make_tensor(
      make_gmem_ptr(param.dv_ptr + dv_offset),
      make_layout(shapeKtV, make_stride(param.dv_r_stride, _1{}, _1{})));
  Tensor mdP = make_tensor(
      make_gmem_ptr(param.pb_ptr + dsb_offset),
      make_layout(shapeSP, make_stride(block_n_dim, _1{}, _1{})));
  Tensor mdQaccum = make_tensor(
      make_gmem_ptr(param.dqaccum_ptr + dqaccum_offset),
      make_layout(shapedQ, make_stride(param.head_dim, _1{}, _1{})));
  Tensor mdK = make_tensor(
      make_gmem_ptr(param.dk_ptr + dk_offset),
      make_layout(shapeKtV, make_stride(param.dk_r_stride, _1{}, _1{})));

  auto tile_sdp = typename Trait::TileShapeSdP{};
  auto tile_dkv = typename Trait::TileShapedKV{};
  auto tile_dq = typename Trait::TileShapedQ{};

  auto tileloadQ = typename Trait::TiledLoadQ{mQ};
  auto tileloadKt = typename Trait::TiledLoadKt{mKt};
  auto tileloaddO = typename Trait::TiledLoaddO{mdO};
  auto tileloadV = typename Trait::TiledLoadV{mV};
  auto tileloadPt = typename Trait::TiledLoadPt{mPt};
  auto tileloaddOt =
      typename Trait::TiledLoaddOt{mdOt}; // load dO as operand B for dV=Pt*dO
  auto tileloaddP = typename Trait::TiledLoaddP{mdP};
  auto tileloadK = typename Trait::TiledLoadK{mK};
  auto tileloaddQ = typename Trait::TiledLoaddQ{mdQaccum};
  auto tileloaddPt = typename Trait::TiledLoaddPt{mdPt};
  auto tileloadQt = typename Trait::TiledLoadQt{mQt};

  auto tilesaveP = typename Trait::TiledSaveS{mP}; // to internal buffer
  auto tilesavedV = typename Trait::TiledSavedV{mdV};
  auto tilesavedP = typename Trait::TiledSavedP{mdP};
  auto tilesavedQ = typename Trait::TiledSavedQ{mdQaccum};
  auto tilesavedK = typename Trait::TiledSavedK{mdK};

  Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
  Tensor mdQ_coord = cute::get_xe_tensor(shapedQ);
  Tensor mKtV_coord = cute::get_xe_tensor(shapeKtV);
  Tensor mdO_coord = cute::get_xe_tensor(shapeO);
  Tensor mQtdOt_coord = cute::get_xe_tensor(shapeQtOt);
  Tensor mK_coord = cute::get_xe_tensor(shapeK);

  Tensor mSP_coord = cute::get_xe_tensor(shapeSP);
  Tensor mPt_coord = cute::get_xe_tensor(shapePt);

  typename Trait::TiledMmaSdP tiled_mma_sdp;
  typename Trait::TiledMmadKV tiled_mma_dkv;
  typename Trait::TiledMmadQ tiled_mma_dq;

  auto thr_mma_sdp = tiled_mma_sdp.get_slice(first_thread_in_sg_idx);
  auto thr_mma_dkv = tiled_mma_dkv.get_slice(first_thread_in_sg_idx);
  auto thr_mma_dq = tiled_mma_dq.get_slice(first_thread_in_sg_idx);

  Tensor gQ = local_tile(mQ_coord, select<0, 2>(tile_sdp), make_coord(_, _, 0));
  Tensor gKtV =
      local_tile(mKtV_coord, select<1, 2>(tile_sdp), make_coord(_, _, 0));
  Tensor gdO =
      local_tile(mdO_coord, select<0, 2>(tile_sdp), make_coord(_, _, 0));
  Tensor gPt = local_tile(
      mPt_coord, select<0, 2>(tile_dkv), make_coord(_, _, 0)); // load Pt
  Tensor gdPa = local_tile(
      mSP_coord, select<0, 2>(tile_dq), make_coord(_, _, 0)); // operand A dQ
  Tensor gK = local_tile(
      mK_coord, select<1, 2>(tile_dq), make_coord(_, _, 0)); // operand B dQ
  Tensor gdPt = local_tile(
      mPt_coord, select<0, 2>(tile_dkv), make_coord(_, _, 0)); // load dpt
  Tensor gQtdOt = local_tile(
      mQtdOt_coord,
      select<1, 2>(tile_dkv),
      make_coord(_, _, 0)); // load Q as operand B

  Tensor gSP = local_tile(
      mSP_coord, select<0, 1>(tile_sdp), make_coord(_, _, 0)); // dump P
  Tensor gdV = local_tile(
      mKtV_coord, select<0, 1>(tile_dkv), make_coord(_, _, 0)); // dump dV
  Tensor gdQ = local_tile(
      mdQ_coord, select<0, 1>(tile_dq), make_coord(_, _, 0)); // dump dQ
  Tensor gdK = local_tile(
      mKtV_coord, select<0, 1>(tile_dkv), make_coord(_, _, 0)); // dump dK

  Tensor tPgP = thr_mma_sdp.partition_C(gSP); // save P to internal buffer
  Tensor tdVgdV = thr_mma_dkv.partition_C(gdV); // save to dv
  Tensor tdQgdQ = thr_mma_dq.partition_C(gdQ); // save to dq
  Tensor tdKgdK = thr_mma_dkv.partition_C(gdK); // save to dk

  // Retile registers for copies

  // Retile global counting tensors for copies

  Tensor tdVrdV = partition_fragment_C(
      tiled_mma_dkv,
      make_shape(
          get<0>(tile_dkv),
          get<1>(tile_dkv),
          ceil_div(Int<kBlockN>{}, get<0>(tile_dkv)),
          ceil_div(Int<kHeadDim>{}, get<1>(tile_dkv))));
  Tensor tdKrdK = partition_fragment_C(
      tiled_mma_dkv,
      make_shape(
          get<0>(tile_dkv),
          get<1>(tile_dkv),
          ceil_div(Int<kBlockN>{}, get<0>(tile_dkv)),
          ceil_div(Int<kHeadDim>{}, get<1>(tile_dkv))));
  // for lse read
  Tensor caccS = make_identity_tensor(
      Shape<Int<kBlockM>, Int<kBlockN>>{}); // same buffer as accS
  Tensor taccScS = thr_mma_sdp.partition_C(caccS);
  static_assert(decltype(size<0>(taccScS))::value == 8);
  // static_assert(size<0>(tSrS) * size<1>(tSrS) == size<0>(lse) && "row of acc
  // and lse not match"); misc

  const int max_m_block = ceil_div(param.seq_len_q, kBlockM);
  const int tail_m = param.seq_len_q % kBlockM;

  cutlass::NumericConverter<T, float> converter;
  // clear accumulator
  clear(tdVrdV);
  clear(tdKrdK);
  for (int m_block = 0; m_block < max_m_block; ++m_block) {
    const bool Is_even_M = not((m_block == max_m_block - 1) and (tail_m != 0));
    if (not Is_even_M) {
      mQ = make_tensor(
          make_gmem_ptr(mQ.data()),
          make_layout(
              make_shape(tail_m, Int<kHeadDim>{}, _1{}),
              make_stride(param.q_r_stride, _1{}, _1{})));
      mdO = make_tensor(
          make_gmem_ptr(mdO.data()),
          make_layout(
              make_shape(tail_m, Int<kHeadDim>{}, _1{}),
              make_stride(param.do_r_stride, _1{}, _1{})));
      mdOt = make_tensor(
          make_gmem_ptr(mdOt.data()),
          make_layout(
              make_shape(Int<kHeadDim>{}, tail_m, _1{}),
              make_stride(_1{}, param.do_r_stride, _1{})));
      mdQaccum = make_tensor(
          make_gmem_ptr(mdQaccum.data()),
          make_layout(shapedQ, make_stride(param.head_dim, _1{}, _1{})));
      mQt = make_tensor(
          make_gmem_ptr(mQt.data()),
          make_layout(
              make_shape(Int<kHeadDim>{}, tail_m, _1{}),
              make_stride(_1{}, param.q_r_stride, _1{})));

      tileloadQ = typename Trait::TiledLoadQ{mQ};
      tileloaddO = typename Trait::TiledLoaddO{mdO};
      tileloaddOt = typename Trait::TiledLoaddOt{mdOt};
      tileloaddQ = typename Trait::TiledLoaddQ{mdQaccum};
      tileloadQt = typename Trait::TiledLoadQt{mQt};
      tilesavedQ = typename Trait::TiledSavedQ{mdQaccum};
    }
    {
      Tensor tSrS = partition_fragment_C(
          tiled_mma_sdp,
          make_shape(
              get<0>(tile_sdp),
              get<1>(tile_sdp),
              ceil_div(Int<kBlockM>{}, get<0>(tile_sdp)),
              ceil_div(Int<kBlockN>{}, get<1>(tile_sdp))));
      clear(tSrS);
      // S=QKt
      gemm_S<T>(
          tSrS,
          gQ,
          gKtV,
          thr_mma_sdp,
          tiled_mma_sdp,
          tile_sdp,
          tileloadQ,
          tileloadKt);
      Tensor scores =
          make_tensor(tSrS.data(), convert_layout_acc_layout(tSrS.layout()));
      Tensor dP_sum =
          softmax_forward<is_causal, Is_even_N, T, V, kBlockM, kBlockN>(
              scores,
              tSrS,
              taccScS,
              mLSE,
              mdPsum,
              tPgP,
              tilesaveP,
              param,
              m_block,
              n_block,
              Is_even_M,
              tail_m);
      auto tSrSl = make_tensor_like<T>(tSrS);
      convert_type(converter, tSrS, tSrSl);
      mha_save<Is_even_N>(tilesaveP, tSrSl, tPgP); // save P to internal buffers
      Tensor tdPrdP = partition_fragment_C(
          tiled_mma_sdp,
          make_shape(
              get<0>(tile_sdp),
              get<1>(tile_sdp),
              ceil_div(Int<kBlockM>{}, get<0>(tile_sdp)),
              ceil_div(Int<kBlockN>{}, get<1>(tile_sdp))));
      clear(tdPrdP);
      // dP=dO*Vt
      gemm_dP<T>(
          tdPrdP,
          gdO,
          gKtV,
          thr_mma_sdp,
          tiled_mma_sdp,
          tile_sdp,
          tileloaddO,
          tileloadV);
      Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
      // dS=P(dP-sum_row(P))*scale
      softmax_backward(scores, dP_sum, dS, param.scale_softmax);
      auto tdPrdPl = make_tensor_like<T>(tdPrdP);
      convert_type(converter, tdPrdP, tdPrdPl);
      mha_save<Is_even_N>(
          tilesavedP, tdPrdPl, tPgP); // save dP to buffer after P used by dV
    }

    // dV=Pt*dO
    gemm_dV<T>(
        tdVrdV,
        gPt,
        gQtdOt,
        thr_mma_dkv,
        tiled_mma_dkv,
        tile_dkv,
        tileloadPt,
        tileloaddOt);
    {
      Tensor tdQrdQ = partition_fragment_C(
          tiled_mma_dq,
          make_shape(
              get<0>(tile_dq),
              get<1>(tile_dq),
              ceil_div(Int<kBlockM>{}, get<0>(tile_dq)),
              ceil_div(Int<kHeadDim>{}, get<1>(tile_dq))));
      clear(tdQrdQ);
      // dQ=dP*K
      gemm_dQ<T>(
          tdQrdQ,
          gdPa,
          gK,
          thr_mma_dq,
          tiled_mma_dq,
          tile_dq,
          tileloaddP,
          tileloadK);
      mha_atomic_add(mdQaccum, tdQgdQ, tdQrdQ, local_id);
    }
    // dK=dPt*Q
    gemm_dK<T>(
        tdKrdK,
        gdPt,
        gQtdOt,
        thr_mma_dkv,
        tiled_mma_dkv,
        tile_dkv,
        tileloaddPt,
        tileloadQt);
    // update ptr/atom copy
    mQ.data() = mQ.data() + int(kBlockM * param.q_r_stride);
    mdO.data() = mdO.data() + int(kBlockM * param.do_r_stride);
    mdOt.data() = mdOt.data() + int(kBlockM * param.do_r_stride);
    mdQaccum.data() = mdQaccum.data() + int(kBlockM * param.head_dim);
    mQt.data() = mQt.data() + int(kBlockM * param.q_r_stride);
    mLSE.data() = mLSE.data() + int(kBlockM);
    mdPsum.data() = mdPsum.data() + int(kBlockM);

    tileloadQ = typename Trait::TiledLoadQ{mQ};
    tileloaddO = typename Trait::TiledLoaddO{mdO};
    tileloaddOt = typename Trait::TiledLoaddOt{mdOt};
    tileloaddQ = typename Trait::TiledLoaddQ{mdQaccum};
    tileloadQt = typename Trait::TiledLoadQt{mQt};
    tilesavedQ = typename Trait::TiledSavedQ{mdQaccum};
  }
  auto tdVrdVl = make_tensor_like<T>(tdVrdV);
  convert_type(converter, tdVrdV, tdVrdVl);
  mha_save<Is_even_N>(tilesavedV, tdVrdVl, tdVgdV);
  auto tdKrdKl = make_tensor_like<T>(tdKrdK);
  convert_type(converter, tdKrdK, tdKrdKl);
  mha_save<Is_even_N>(tilesavedK, tdKrdKl, tdKgdK);
}

template <class T>
void mha_backward_seq(T trait, Param<typename T::DType> param) {
  const int bidb = BlockIdxZ();
  const int bidhq = BlockIdxY();
  const int bidnblk = BlockIdxX();
  const int bidhkv = bidhq / param.num_qh_per_kvh;
  for (int n_block = bidnblk; n_block < param.n_block; n_block += GridDimX()) {
    if (param.tail_n > 0 and n_block == param.n_block - 1)
      dq_dk_dv_1colblock<false, false>(
          trait, param, bidb, bidhq, bidhkv, param.n_block - 1, param.tail_n);
    else
      dq_dk_dv_1colblock<true, false>(
          trait, param, bidb, bidhq, bidhkv, n_block);
  }
}

template <class T>
void mha_backward_parallel(T trait, Param<typename T::DType> param) {
  const int bidb = BlockIdxZ();
  const int bidhq = BlockIdxY();
  const int n_block = BlockIdxX();
  const int bidhkv = bidhq / param.num_qh_per_kvh;
  if (param.tail_n > 0 and n_block == param.n_block - 1)
    dq_dk_dv_1colblock<false, true>(
        trait, param, bidb, bidhq, bidhkv, param.n_block - 1, param.tail_n);
  else
    dq_dk_dv_1colblock<true, true>(trait, param, bidb, bidhq, bidhkv, n_block);
}

template <bool Is_even_M, class T>
void convert_dq(
    T& trait,
    Param<typename T::DType>& param,
    int m_block,
    int bidb,
    int bidh) {
  constexpr int kBlockM = T::kBlockM;
  constexpr int kHeadDim = T::kHeadDim;
  using DType = typename T::DType;
  auto sg = compat::get_nd_item<1>().get_sub_group();
  auto first_thread_in_sg_idx = sg.get_group_linear_id() * trait.SubgroupSize;

  auto bofst = Boffset(param);
  const index_t dqaccum_offset =
      bofst.dqaccum_offset(bidb, bidh, m_block * kBlockM);
  const index_t dq_offset = bofst.dq_offset(bidb, bidh, m_block * kBlockM);
  using ShapeQ = Shape<
      std::conditional_t<Is_even_M, Int<kBlockM>, int>,
      Int<kHeadDim>,
      _1>;
  ShapeQ shapeQ;
  if constexpr (Is_even_M) {
    shapeQ = make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, _1{});
  } else {
    shapeQ = make_shape(param.tail_m, Int<kHeadDim>{}, _1{});
  }

  Tensor mdQaccum = make_tensor(
      make_gmem_ptr(param.dqaccum_ptr + dqaccum_offset),
      make_layout(
          Shape<Int<kBlockM>, Int<kHeadDim>>{},
          make_stride(param.head_dim, _1{}, _1{})));
  Tensor mdQ = make_tensor(
      make_gmem_ptr(param.dq_ptr + dq_offset),
      make_layout(shapeQ, make_stride(param.dq_r_stride, _1{}, _1{})));

  auto tile_dq = typename T::TileShapedQ{};

  auto tileloaddQ = typename T::TiledLoaddQ{mdQaccum};
  auto tilesavedQ = typename T::TiledSavedV{mdQ};

  typename T::TiledMmadQ tiled_mma_dq;
  auto thr_mma_dq = tiled_mma_dq.get_slice(first_thread_in_sg_idx);

  Tensor mQ_coord = cute::get_xe_tensor(shapeQ);
  Tensor gdQ = local_tile(
      mQ_coord, select<0, 1>(tile_dq), make_coord(_, _, 0)); // dump dQ

  Tensor tdQgdQ = thr_mma_dq.partition_C(gdQ); // save to dq
  Tensor tdQrdQaccum = partition_fragment_C(
      tiled_mma_dq,
      make_shape(
          get<0>(tile_dq),
          get<1>(tile_dq),
          ceil_div(Int<kBlockM>{}, get<0>(tile_dq)),
          ceil_div(Int<kHeadDim>{}, get<1>(tile_dq))));

  Tensor tdQrdQ = make_fragment_like<DType>(tdQrdQaccum);
  if constexpr (Is_even_M) {
    mha_load<true>(tileloaddQ, tdQgdQ, tdQrdQaccum);
  } else {
    mha_load<false>(tileloaddQ, tdQgdQ, tdQrdQaccum);
  }
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(tdQrdQ); ++i) {
    tdQrdQ(i) = static_cast<DType>(tdQrdQaccum(i));
  }
  if constexpr (Is_even_M) {
    mha_save<true>(tilesavedQ, tdQrdQ, tdQgdQ);
  } else {
    mha_save<false>(tilesavedQ, tdQrdQ, tdQgdQ);
  }
}

template <class T>
void mhd_convert_dq(T trait, Param<typename T::DType> param) {
  // The block index for the M dimension.
  const int m_block = BlockIdxX();
  // The block index for the batch.
  const int bidb = BlockIdxZ();
  // The block index for the head.
  const int bidh = BlockIdxY();
  if (param.tail_m > 0 and m_block == param.m_block - 1) {
    convert_dq<false>(trait, param, m_block, bidb, bidh);
  } else {
    convert_dq<true>(trait, param, m_block, bidb, bidh);
  }
}

template <class...>
class MhaDotDoOName;

template <class...>
class MhaBackwardName;

template <class...>
class MhdConvertDqName;

template <
    typename T,
    int kBlockM,
    int kBlockN,
    int kHeadDim,
    int kNSGs,
    int AtomLayoutMSdP,
    int AtomLayoutNdKV,
    int AtomLayoutMdQ,
    bool is_causal>
void run_mha_bwd_specialized(
    sycl::queue& queue,
    FLASH_BWD_params& flash_bwd_params) {
  auto trait = FAKernel<
      T,
      kHeadDim,
      kBlockM,
      kBlockN,
      kNSGs,
      AtomLayoutMSdP,
      AtomLayoutNdKV,
      AtomLayoutMdQ,
      is_causal>{};

  const int BATCH = flash_bwd_params.batch_size;
  const int NUM_HEAD_Q = flash_bwd_params.num_heads_qo;
  const int NUM_HEAD_KV = flash_bwd_params.num_heads_kv;
  const int SEQ_LEN_Q = flash_bwd_params.seqlen_qo;
  const int SEQ_LEN_KV = flash_bwd_params.seqlen_kv;
  const int N_BLOCK = ceil_div(SEQ_LEN_KV, kBlockN);
  const int tail_n = SEQ_LEN_KV % kBlockN;
  const int M_BLOCK = ceil_div(SEQ_LEN_Q, kBlockM);
  const int tail_m = SEQ_LEN_Q % kBlockM;
  auto param = Param<T>(
      static_cast<const T*>(flash_bwd_params.do_ptr),
      static_cast<const T*>(flash_bwd_params.o_ptr),
      static_cast<const T*>(flash_bwd_params.q_ptr),
      static_cast<const T*>(flash_bwd_params.k_ptr),
      static_cast<const T*>(flash_bwd_params.v_ptr),
      static_cast<const float*>(flash_bwd_params.lse_ptr),
      static_cast<float*>(flash_bwd_params.odo_ptr),
      static_cast<float*>(flash_bwd_params.dqaccum_ptr),
      static_cast<T*>(flash_bwd_params.dq_ptr),
      static_cast<T*>(flash_bwd_params.dk_ptr),
      static_cast<T*>(flash_bwd_params.dv_ptr),
      static_cast<T*>(flash_bwd_params.pb_ptr),
      flash_bwd_params.scale);
  param.batch = BATCH;
  param.num_head_q = NUM_HEAD_Q;
  param.num_head_kv = NUM_HEAD_KV;
  param.num_qh_per_kvh = NUM_HEAD_Q / NUM_HEAD_KV;
  param.num_nb_per_blk = std::max(
      N_BLOCK * NUM_HEAD_Q * BATCH / 1024,
      1); // 1024 tuneable here, best for pvc
  param.seq_len_q = SEQ_LEN_Q;
  param.seq_len_kv = SEQ_LEN_KV;
  param.head_dim = kHeadDim;
  param.n_block = N_BLOCK;
  param.tail_n = tail_n;
  param.m_block = M_BLOCK;
  param.tail_m = tail_m;
  param.seq_len_kv_pad = flash_bwd_params.seqlen_kv_pad;
  param.seq_len_q_pad = flash_bwd_params.seqlen_qo_pad;

  setup_stride(param, flash_bwd_params);

  auto dimGrid0 =
      compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
  auto dimBlock0 =
      compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
  compat::experimental::launch_properties launch_props0{};
  compat::experimental::kernel_properties kernel_props0{
      sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
  compat::experimental::launch_policy policy0{
      dimGrid0, dimBlock0, launch_props0, kernel_props0};
  compat::experimental::
      launch<mha_dot_do_o<decltype(trait)>, MhaDotDoOName<decltype(trait)>>(
          policy0, queue, trait, param);

  auto dimGrid1 = compat::dim3(
      size(ceil_div(param.n_block, param.num_nb_per_blk)),
      size(param.num_head_q),
      size(param.batch));
  auto dimBlock1 =
      compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
  compat::experimental::launch_properties launch_props1{
      sycl::ext::oneapi::experimental::work_group_scratch_size(
          trait.smem_size)};
  compat::experimental::kernel_properties kernel_props1{
      sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
  compat::experimental::launch_policy policy1{
      dimGrid1, dimBlock1, launch_props1, kernel_props1};
  compat::experimental::launch<
      mha_backward_seq<decltype(trait)>,
      MhaBackwardName<decltype(trait)>>(policy1, queue, trait, param);

  auto dimGrid2 =
      compat::dim3(size(M_BLOCK), size(param.num_head_q), size(param.batch));
  auto dimBlock2 =
      compat::dim3(size(kNSGs * trait.SubgroupSize), size(1), size(1));
  compat::experimental::launch_properties launch_props2{};
  compat::experimental::kernel_properties kernel_props2{
      sycl::ext::oneapi::experimental::sub_group_size<trait.SubgroupSize>};
  compat::experimental::launch_policy policy2{
      dimGrid2, dimBlock2, launch_props2, kernel_props2};
  auto event2 = compat::experimental::launch<
      mhd_convert_dq<decltype(trait)>,
      MhdConvertDqName<decltype(trait)>>(policy2, queue, trait, param);
}

template <typename T, int kMPad, int kNPad, bool is_causal>
void run_mha_bwd_(sycl::queue& queue, FLASH_BWD_params& params) {
  const int headdim = params.head_size_vo;
#define RUN_MHA_BWD_SPECIALIZED() \
  run_mha_bwd_specialized<        \
      T,                          \
      kBlockM,                    \
      kBlockN,                    \
      kHeadDim,                   \
      kNSGs,                      \
      AtomLayoutMSdP,             \
      AtomLayoutNdKV,             \
      AtomLayoutMdQ,              \
      is_causal>(queue, params);

  if (headdim == 64) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 32;
    constexpr int kHeadDim = 64;
    constexpr int kNSGs = 8;
    constexpr int AtomLayoutMSdP = 4;
    constexpr int AtomLayoutNdKV = 2;
    constexpr int AtomLayoutMdQ = 2;
    static_assert(
        kBlockM <= kMPad, "kBlockM must be less than or equal to kMPad");
    static_assert(
        kBlockN <= kNPad, "kBlockN must be less than or equal to kNPad");
    RUN_MHA_BWD_SPECIALIZED();
  } else if (headdim == 96) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kHeadDim = 96;
    constexpr int kNSGs = 8;
    constexpr int AtomLayoutMSdP = 2;
    constexpr int AtomLayoutNdKV = 4;
    constexpr int AtomLayoutMdQ = 4;
    static_assert(
        kBlockM <= kMPad, "kBlockM must be less than or equal to kMPad");
    static_assert(
        kBlockN <= kNPad, "kBlockN must be less than or equal to kNPad");
    RUN_MHA_BWD_SPECIALIZED();
  } else if (headdim == 128) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kHeadDim = 128;
    constexpr int kNSGs = 8;
    constexpr int AtomLayoutMSdP = 4;
    constexpr int AtomLayoutNdKV = 4;
    constexpr int AtomLayoutMdQ = 4;
    static_assert(
        kBlockM <= kMPad, "kBlockM must be less than or equal to kMPad");
    static_assert(
        kBlockN <= kNPad, "kBlockN must be less than or equal to kNPad");
    RUN_MHA_BWD_SPECIALIZED();
  } else if (headdim == 192) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 32;
    constexpr int kHeadDim = 192;
    constexpr int kNSGs = 8;
    constexpr int AtomLayoutMSdP = 4;
    constexpr int AtomLayoutNdKV = 2;
    constexpr int AtomLayoutMdQ = 2;
    static_assert(
        kBlockM <= kMPad, "kBlockM must be less than or equal to kMPad");
    static_assert(
        kBlockN <= kNPad, "kBlockN must be less than or equal to kNPad");
    RUN_MHA_BWD_SPECIALIZED();
  } else if (headdim == 256) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 32;
    constexpr int kHeadDim = 256;
    constexpr int kNSGs = 8;
    constexpr int AtomLayoutMSdP = 4;
    constexpr int AtomLayoutNdKV = 2;
    constexpr int AtomLayoutMdQ = 2;
    static_assert(
        kBlockM <= kMPad, "kBlockM must be less than or equal to kMPad");
    static_assert(
        kBlockN <= kNPad, "kBlockN must be less than or equal to kNPad");
    RUN_MHA_BWD_SPECIALIZED();
  } else {
    TORCH_CHECK(
        false,
        "FlashAttentionBackwardXPU: unsupported head dimension: ",
        headdim);
  }
#undef RUN_MHA_BWD_SPECIALIZED
}

template <int kMPad, int kNPad>
void run_mha_bwd(sycl::queue& queue, FLASH_BWD_params& params) {
  FP16_SWITCH(params.is_fp16, [&] {
    BOOL_SWITCH(params.is_causal, IS_CAUSAL, [&] {
      run_mha_bwd_<elem_type, kMPad, kNPad, IS_CAUSAL>(queue, params);
    });
  });
}

} // namespace cute

namespace sycltla {

std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_attention_backward_sycltla(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    const double dropout,
    const bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const float scale) {
  TORCH_CHECK(
      dropout == 0.0,
      "FlashAttentionBackwardXPU does not only support dropout > 0.0 yet");

  at::Tensor contiguous_grad_out = grad_out.contiguous();
  CHECK_DEVICE(query);
  CHECK_DEVICE(key);
  CHECK_DEVICE(value);
  CHECK_DEVICE(out);
  CHECK_DEVICE(contiguous_grad_out);
  CHECK_DEVICE(logsumexp);

  TORCH_CHECK(
      !query.is_nested() && !key.is_nested() && !value.is_nested() &&
          !out.is_nested() && !grad_out.is_nested() && !logsumexp.is_nested(),
      "FlashAttentionBackwardXPU only support dense inputs");

  auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kBFloat16,
      "FlashAttentionBackwardXPU only support fp16 and bf16 data type");
  TORCH_CHECK(
      logsumexp.scalar_type() == at::kFloat,
      "FlashAttentionBackwardXPU: logsumexp must have the dtype float32");
  TORCH_CHECK(
      key.scalar_type() == dtype,
      "FlashAttentionBackwardXPU: query and key must have the same dtype");
  TORCH_CHECK(
      value.scalar_type() == dtype,
      "FlashAttentionBackwardXPU: query and value must have the same dtype");
  TORCH_CHECK(
      out.scalar_type() == dtype,
      "FlashAttentionBackwardXPU: query and out must have the same dtype");
  TORCH_CHECK(
      contiguous_grad_out.scalar_type() == dtype,
      "FlashAttentionBackwardXPU: query and grad_out must have the same dtype");

  TORCH_CHECK(
      query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
          out.dim() == 4 && contiguous_grad_out.dim() == 4 &&
          logsumexp.dim() == 3,
      "FlashAttentionBackwardXPU requires query, key, value, out, grad_out to be 4 dimensional and logsumexp to be 3 dimensional");

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
  CHECK_SHAPE(out, batch_size, numhead_qo, seqlen_qo, headsize_vo);
  CHECK_SHAPE(
      contiguous_grad_out, batch_size, numhead_qo, seqlen_qo, headsize_vo);
  CHECK_SHAPE(logsumexp, batch_size, numhead_qo, seqlen_qo);
  TORCH_CHECK(
      numhead_qo % numhead_kv == 0,
      "FlashAttentionBackwardXPU: number of heads in key/value must divide number of heads in query");
  TORCH_CHECK(
      headsize_qk == headsize_vo,
      "FlashAttentionBackwardXPU: headsize_qk must be equal to headsize_vo");

  TORCH_CHECK(
      query.stride(-1) == 1,
      "FlashAttentionBackwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      key.stride(-1) == 1,
      "FlashAttentionBackwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      value.stride(-1) == 1,
      "FlashAttentionBackwardXPU: input tensor must have contiguous last dimension");
  TORCH_CHECK(
      out.stride(-1) == 1,
      "FlashAttentionBackwardXPU: out tensor must have contiguous last dimension");
  TORCH_CHECK(
      contiguous_grad_out.stride(-1) == 1,
      "FlashAttentionBackwardXPU: dout tensor must have contiguous last dimension");
  TORCH_CHECK(
      logsumexp.stride(-1) == 1,
      "FlashAttentionBackwardXPU: logsumexp tensor must have contiguous last dimension");

  TORCH_CHECK(
      logsumexp.is_contiguous(),
      "FlashAttentionBackwardXPU: logsumexp must be contiguous in [batch_size, numhead_qo, seqlen_qo]");

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
        "XPU device architecture does not support flash attention backward. Supported architectures are: intel_gpu_pvc, intel_gpu_pvc_vg, intel_gpu_bmg_g21.");
  }

  auto grad_query = at::empty_like(query);
  auto grad_key = at::empty_like(key);
  auto grad_value = at::empty_like(value);

  auto opts = query.options();

  at::Tensor grad_key_expanded, grad_value_expanded;
  if (numhead_kv != numhead_qo) { // MQA / GQA
    grad_key_expanded =
        at::empty({batch_size, numhead_qo, seqlen_kv, headsize_qk}, opts);
    grad_value_expanded =
        at::empty({batch_size, numhead_qo, seqlen_kv, headsize_vo}, opts);
  } else {
    grad_key_expanded = grad_key;
    grad_value_expanded = grad_value;
  }

  constexpr int kMPad = 128;
  constexpr int kNPad = 128;
  int seqlen_qo_pad = (seqlen_qo + kMPad - 1) / kMPad * kMPad;
  int seqlen_kv_pad = (seqlen_kv + kNPad - 1) / kNPad * kNPad;
  auto tensor_odo = at::empty_like(out, opts.dtype(at::kFloat));
  auto tensor_dqaccum = at::empty(
      {batch_size, numhead_qo, seqlen_qo_pad, headsize_qk},
      opts.dtype(at::kFloat));
  auto tensor_pbuff =
      at::empty({batch_size, numhead_qo, seqlen_kv_pad, 2 * kMPad}, opts);

  FLASH_BWD_params params;
  set_params_dgrad(
      params,
      batch_size,
      numhead_qo,
      numhead_kv,
      seqlen_qo,
      seqlen_kv,
      headsize_qk,
      headsize_vo,
      seqlen_qo_pad,
      seqlen_kv_pad,
      query,
      key,
      value,
      out,
      contiguous_grad_out,
      logsumexp,
      grad_query,
      grad_key_expanded,
      grad_value_expanded,
      tensor_odo,
      tensor_dqaccum,
      tensor_pbuff,
      scale,
      is_causal);

  cute::run_mha_bwd<kMPad, kNPad>(sycl_queue, params);

  if (numhead_kv != numhead_qo) {
    at::sum_out(
        grad_key,
        at::reshape(
            grad_key_expanded,
            {batch_size,
             numhead_kv,
             numhead_qo / numhead_kv,
             seqlen_kv,
             headsize_qk}),
        {2});
    at::sum_out(
        grad_value,
        at::reshape(
            grad_value_expanded,
            {batch_size,
             numhead_kv,
             numhead_qo / numhead_kv,
             seqlen_kv,
             headsize_vo}),
        {2});
  }

  return std::make_tuple(
      std::move(grad_query), std::move(grad_key), std::move(grad_value));
}
} // namespace sycltla
