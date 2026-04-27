/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/epilogue/collective/collective_epilogue.hpp>
#include <cutlass/epilogue/collective/detail.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>

#include <cute/algorithm/subgroup_algorithms.hpp>
#include <cute/algorithm/tensor_algorithms.hpp>

#include <flash_attention_v2/collective/copy_block_slm.hpp>
#include <flash_attention_v2/collective/fmha_fusion.hpp>
#include <sycl/sycl.hpp>

namespace cutlass::fmha::collective {

using namespace cute;

template <
    class CollectiveMainloop, // Attention mainloop
    class TileShapeO_, // Shape of output tile, may be larger than P*V GEMM
    class TensorO_, // 2D slice of global output tensor
    class TiledCopyO_ = void> // Optional TiledCopy for loading O
class FMHAFwdEpilogue {
 public:
  //
  // Type Aliases
  //
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D =
      decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_, _), 0)));
  using ElementO = typename TensorO_::value_type;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;
  using ElementA = typename FragA::value_type;

  // Split k-reduced tiles between participating subgroups.
  // Assumption: the A tile is contiguous.
  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int < (v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg)
                                           : v_total_sg > {};
  }

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));
  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout =
      decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));

  using SGTileShapeO =
      decltype(shape_div(take<0, 2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
      make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  static auto default_tiled_copy_O_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(
          TiledMMAPV{},
          ReduceFragA{}.tv_layout(),
          ReduceSGLayout{},
          TensorO2D{});
  }

  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO =
      conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  // Stateless design -- no arguments or parameters.
  struct Arguments {};
  struct Params {};

  // Shared memory storage
  // Note sum/max tiles are padded to 16 elements, due to limitations in CuTe
  // block load infrastructure.
  using AlignedSGTileA_Q =
      C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) *
        intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data,
        a_max_data;
  };

  using SharedStorage = conditional_t<
      (ReduceK{} > _1{}),
      SharedStorageReduceK,
      SharedStorageNone>;

 private:
  SharedStorage& shared;

 public:
  static constexpr Params to_underlying_arguments(
      Arguments const& args,
      void* /* workspace */) {
    return {};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FMHAFwdEpilogue(Params const&, SharedStorage& shared_) : shared(shared_) {}

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O, // Global O tensor: (q,v)
      FragA& tArA, // O accumulator:   (q,v)
      FragARow& tA_max, // Softmax row-wise max accumulator
      FragARow& tA_sum, // Softmax row-wise sum accumulator
      QVCoord blk_qv, // WG tile indices: (q,v)
      int thr_id, // Work-item ID
      float* pLSE, // Global LSE Ptr
      const std::tuple<int, int, int, int, int, int, int>&
          metadata_for_lse // Metadata for LSE to calculate offset
  ) {
    using namespace cute;
    using ElementA = typename FragA::element_type;

    // Reduce k-blocks of A and A_sum across WG, if needed.
    auto [rA, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    /* Some subgroups may not have any work to do; if so, quit early. */
    if (!active)
      return;

    auto non_reciprocal_rAsum = rA_sum(0);
    /* Complete softmax, dividing out sums. */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++)
      rA_sum(i) = ElementA(1) / rA_sum(i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++) {
      rA(i) *= broadcast<0>(rA_sum, rA, i);
      if (std::isnan(rA(i))) {
        rA(i) =
            0; // Handle the -nan when the whole sequence is completely masked
      }
    }

    /* Tile output */
    Tensor cO = make_identity_tensor(O.shape()); // (q,v)
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv); // (q,v)

    /* Prepare slices */
    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);

    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    /* Reorder tile and write out */
    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);

    /* Calculate the LSE*/
    auto
        [blk_q,
         num_heads_q,
         seq_len_qo,
         batch_idx,
         q_head_idx,
         tile_row_idx,
         rows_of_maxima] = metadata_for_lse;
    int blk_q_coord = get<0>(blk_qv);
    size_t lse_offset =
        batch_idx * num_heads_q * seq_len_qo + // shift the batch
        q_head_idx * seq_len_qo + // shift the head
        blk_q_coord * blk_q; // shift to the particular tile
    size_t seq_coord = blk_q_coord * blk_q + tile_row_idx;

    // There is an implicit mapping that lane_id 0 will map to the first row
    // maxima
    auto sg = compat::get_nd_item<1>().get_sub_group();
    int lane_id = static_cast<int>(sg.get_local_linear_id());

    if (tile_row_idx != -1 && seq_coord < seq_len_qo &&
        (tile_row_idx % rows_of_maxima) ==
            lane_id) { // only 1 lane contain the correct row maxima for that
                       // particular row
      // The softmax scale was multiplied by the kLog2e in the mainloop
      // Need to divide it to restore the value
      double kLog2e = 1.4426950408889634074;
      tA_max[0] = tA_max[0] / kLog2e;
      float lse_val = tA_max[0] + logf(non_reciprocal_rAsum);
      *(pLSE + lse_offset + tile_row_idx) = lse_val == -INFINITY ? 0 : lse_val;
    }
  }

  // Reduce k-blocks of A and A_sum across WG, if needed.
  // Note that each k block has its own scale factor based on A_max,
  //   so A/A_sum contributions need to be rescaled to match.
  template <typename FragA, typename FragARow>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA& tArA, // O accumulator:   (q,v)
      FragARow& tA_max, // Softmax row-wise max accumulator
      FragARow& tA_sum, // Softmax row-wise sum accumulator
      int thr_id) { // Work-item ID

    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (ReduceK{} == _1{}) {
      return std::make_tuple(tArA, tA_sum, true);
    } else {
      /* Identify A tile ID and k block for this subgroup. */
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                         .get_flat_coord(assert_uniform(
                             thr_id)); // compress VMNK to V(MN)K. Use the
                                       // threadID to get the value.
      auto a_tile = get<1>(thr_vak); // m,n
      auto k_blk = get<2>(thr_vak); // k

      /* Set up SLM tensors and partition A tiles among participating subgroups
       */
      auto shape_A = append(
          append(SGTileShapeA{}, ReduceK{}),
          SGPerWG{} / ReduceK{}); // M, K, L/stage for pipelining --
      auto shape_A_row = make_shape(
          get<0>(SGTileShapeO{}),
          shape(ReduceSGLayout{}),
          ReduceK{},
          SGPerWG{} / ReduceK{}); // M, N, K, L

      /* Physical layouts, with subtile modes broken out */
      auto sA_layout = group<2, 4>(flat_divide(
          make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}),
          SGTileShapeO{}));
      auto sA_row_stride = make_stride(
          _1{},
          make_stride(get<0>(shape_A_row), _0{}),
          AlignedSGTileA_Q{},
          AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      /* Coordinate layouts, with subtile modes broken out */
      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(
          append(SGTileShapeO{}, shape(ReduceSGLayout{})),
          append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_data),
          sA_layout); // (q,v,rblk_dst,rblk_src,a_tile)
      auto sA_max = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_max_data),
          sA_row_layout); // (q,rblk_dst,rblk_src,a_tile)
      auto sA_sum = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_sum_data),
          sA_row_layout); // (q,rblk_dst,rblk_src,a_tile)

      /* Write my contributions to SLM. */
      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) ||
          (ReduceK{} == size(ReduceSGLayout{})); // help compiler out

      /* Wait for maxima to be available, signal other data available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum, rA_max, rA_kmax[ReduceK{}];

      if (active) {
        /* Read A_max back from SLM and reduce. */
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
        }

        rA_max = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++)
          cute::transform(
              rA_max,
              rA_kmax[kr],
              rA_max,
              cute::max_fn{}); // Finding the max from the K-block

        /* Calculate scale factors for aligning per-block maxima. */
        for (int kr = 0; kr < ReduceK{}; kr++) {
          cute::transform(
              rA_max, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) {
                return sycl::native::exp2(kmax - gmax);
              });
        }
      }

      /* Wait for A/A_sum data to be available */
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        /* Read A/A_sum back from SLM, align scaling to new maxima, and reduce.
         */
        clear(rA_sum);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragARow rA_sum_read;
          copy_block_s2r(sA_sum(_, k_blk, kr, a_tile), rA_sum_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_sum_read.size(); i++) {
            rA_sum(i) += rA_sum_read(i) * rA_kmax[kr](i);
          }
        }

        clear(rA);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragA rA_read;
          copy_block_s2r(
              sA(_, _, k_blk, kr, a_tile), sA_coords(_, _, 0), rA_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_read.size(); i++) {
            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
          }
        }
      }
      return std::make_tuple(rA, rA_sum, active);
    }
  }
};

} // namespace cutlass::fmha::collective
