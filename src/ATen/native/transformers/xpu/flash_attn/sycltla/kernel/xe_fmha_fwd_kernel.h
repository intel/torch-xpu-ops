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

#include <cute/util/type_traits.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/gemm.h>
#include <cutlass/kernel_hardware_info.hpp>
#include <flash_attention_v2/collective/copy_block_slm.hpp>
#include <flash_attention_v2/collective/fmha_fusion.hpp>

#include <sycltla/collective/xe_fmha_fwd_epilogue.h>
#include <sycltla/collective/xe_fmha_fwd_mainloop.h>
#include <sycltla/kernel/xe_tile_scheduler.h>

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <bool IsVarLen_ = false>
struct FMHAProblemShape {
  using SeqLenType = cute::
      conditional_t<IsVarLen_, cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv;
  int head_size_qk, head_size_vo;
};

///////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class XeFMHAFwdKernel {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len =
      cutlass::fmha::collective::is_variable_length_v<
          typename ProblemShape::SeqLenType>;
  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    const int64_t q_batch_stride;
    const int64_t q_head_stride;
    const int64_t q_row_stride;
    const ElementK* K;
    const int64_t k_batch_stride;
    const int64_t k_head_stride;
    const int64_t k_row_stride;
    const ElementV* V;
    const int64_t v_batch_stride;
    const int64_t v_head_stride;
    const int64_t v_row_stride;
    ElementO* O;
    const int64_t o_batch_stride;
    const int64_t o_head_stride;
    const int64_t o_row_stride;
    float* pLSE;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  static Params to_underlying_arguments(
      Arguments const& args,
      void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) &&
        CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(
        params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(
      ProblemShape const& problem_shape,
      int const& batch) {
    if constexpr (is_var_len) {
      return cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength, VariableLength>{
              problem_shape.seq_len_qo, problem_shape.seq_len_kv},
          batch);
    } else {
      return Shape<int, int>{
          problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  // Find the length of the longest non masked sequence within that subgroup
  CUTLASS_DEVICE
  int calculate_longest_non_masked_length(
      const int& seq_len_kv,
      const int& seq_len_qo,
      const int& last_seq_coord,
      const int& first_non_masked_sequence) {
    int longest_non_masked_length = 0;

    if (seq_len_kv > seq_len_qo) {
      // Find out how many elements have to be calculated in the first sequence
      int elements_in_first_line = seq_len_kv - (seq_len_qo - 1);
      longest_non_masked_length = elements_in_first_line +
          last_seq_coord; // the number of elements increased with the sequence
                          // row number.
    }
    if (seq_len_qo > seq_len_kv) {
      longest_non_masked_length = cute::min(
          seq_len_kv,
          cute::max(0, last_seq_coord - first_non_masked_sequence + 1));
    }
    if (seq_len_qo == seq_len_kv) {
      longest_non_masked_length = cute::min(
          seq_len_kv,
          cute::max(0, last_seq_coord - first_non_masked_sequence + 1));
    }

    longest_non_masked_length =
        cute::min(seq_len_kv, longest_non_masked_length);
    return longest_non_masked_length;
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(
        sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] =
          tile_scheduler.get_block_coord(); // (Q,V,h,b)
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = head_q / head_group_q;

      auto [seq_len_qo, seq_len_kv] = get_sequence_length_shape(s, idx_b);
      if (blk_q * get<0>(TileShapeQK{}) >= seq_len_qo)
        continue;

      int seq_coord =
          cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));
      int first_non_masked_sequence = seq_len_qo - seq_len_kv;
      int last_seq_coord = seq_coord + q_sg_tile - 1;

      // Optimization - Skip computations as this current block will not affect
      // the output
      if (CollectiveMainloop::CausalMask &&
          first_non_masked_sequence > last_seq_coord) {
        continue;
      }

      const int seq_len = CollectiveMainloop::CausalMask
          ? calculate_longest_non_masked_length(
                seq_len_kv,
                seq_len_qo,
                last_seq_coord,
                first_non_masked_sequence)
          : seq_len_kv;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        auto kv_cumulative = s.seq_len_kv.cumulative_length;
        offset_q =
            p.q_head_stride * head_q + p.q_row_stride * qo_cumulative[idx_b];
        offset_k =
            p.k_head_stride * head + p.k_row_stride * kv_cumulative[idx_b];
        offset_v =
            p.v_head_stride * head + p.v_row_stride * kv_cumulative[idx_b];
        offset_o =
            p.o_head_stride * head_q + p.o_row_stride * qo_cumulative[idx_b];
      }

      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q =
          make_shape(seq_len_qo, s.head_size_qk, s.num_heads_q, batch_dim);
      auto shape_K =
          make_shape(seq_len_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V =
          make_shape(s.head_size_vo, seq_len_kv, s.num_heads_kv, batch_dim);
      auto shape_O =
          make_shape(seq_len_qo, s.head_size_vo, s.num_heads_q, batch_dim);

      auto dcQ = p.Q + offset_q;
      auto dcK = p.K + offset_k;
      auto dcV = p.V + offset_v;
      auto ptrO = p.O + offset_o;
      auto dpLSE = p.pLSE;

      auto stride_q = cutlass::make_stride(
          p.q_row_stride, Int<1>{}, p.q_head_stride, p.q_batch_stride);
      auto stride_k = cutlass::make_stride(
          p.k_row_stride, Int<1>{}, p.k_head_stride, p.k_batch_stride);
      auto stride_v = cutlass::make_stride(
          Int<1>{}, p.v_row_stride, p.v_head_stride, p.v_batch_stride);
      auto stride_o = cutlass::make_stride(
          p.o_row_stride, Int<1>{}, p.o_head_stride, p.o_batch_stride);

      Tensor Q =
          make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, stride_q));
      Tensor K =
          make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, stride_k));
      Tensor V =
          make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, stride_v));
      Tensor O =
          make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;
      int tile_row_idx = -1;
      int rows_of_maxima =
          get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

      // Main loop
      int l_coord = is_var_len ? 0 : idx_b;
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(
          Q(_, _, head_q, l_coord),
          K(_, _, head, l_coord),
          V(_, _, head, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          0,
          k_blocks,
          k_blocks,
          thr_id,
          seq_len,
          seq_len_qo,
          seq_len_kv,
          idx_b,
          tile_row_idx,
          rows_of_maxima);

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      auto metadata_for_lse = std::make_tuple(
          get<0>(TileShapePV{}),
          s.num_heads_q,
          seq_len_qo,
          idx_b,
          head_q,
          tile_row_idx,
          rows_of_maxima);
      epilogue(
          O(_, _, head_q, l_coord),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          thr_id,
          dpLSE,
          metadata_for_lse);
    }
  }
};

} // namespace cutlass::fmha::kernel
