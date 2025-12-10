/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::flash_attention {

namespace kernel {

struct XeFlashIndividualTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFlashIndividualTileScheduler(Params const& params) : params(params) {}

  template <class ProblemSize, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      bool const& is_bshd) {
    using namespace cute;

    if (is_bshd) {
      dim3 grid(
          size(ceil_div(
              shape<3>(problem_size),
              shape<0>(tile_shape))), // seq_len_qo / 128
          size(shape<1>(problem_size)), // num_heads_q
          size(shape<0>(problem_size))); // batch
      return Params{grid, 1}; // 1 since it will not be used.
    } else {
      // problem_size = [batch, num_heads_q, num_heads_kv, seq_len_qo,
      // seq_len_kv, head_size_qk, head_size_vo]
      dim3 grid(
          size(ceil_div(shape<6>(problem_size), shape<1>(tile_shape))),
          size(ceil_div(shape<3>(problem_size), shape<0>(tile_shape))),
          size(shape<0>(problem_size) * shape<1>(problem_size)));
      return Params{grid, {shape<1>(problem_size)}};
    }
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return valid_;
  }

  CUTLASS_DEVICE
  auto get_block_coord_bshd() {
    using namespace cute;
    return make_coord(BlockIdxX(), BlockIdxY(), BlockIdxZ());
  }

  CUTLASS_DEVICE
  auto get_block_coord_bhsd() {
    using namespace cute;
    int block_decode = BlockIdxZ();
    int bidh;
    params.divmod_num_heads(block_decode, bidh, block_decode);
    return make_coord(BlockIdxX(), BlockIdxY(), block_decode, bidh);
  }

  CUTLASS_DEVICE
  XeFlashIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

////////////////////////////////////////////////////////////////////////////////
} // namespace kernel

struct IndividualScheduler {};

namespace detail {

template <class TileSchedulerTag, class ArchTag, class Enable = void>
struct TileSchedulerSelector {
  static_assert(
      cutlass::detail::dependent_false<ArchTag>,
      "Could not select a tile scheduler for given parameters.");
};

// Default (void) maps to XeFlashIndividualTileScheduler
template <class ArchTag>
struct TileSchedulerSelector<
    void,
    ArchTag,
    cute::enable_if_t<cute::is_same_v<ArchTag, cutlass::arch::IntelXe>>> {
  using Scheduler =
      typename TileSchedulerSelector<IndividualScheduler, ArchTag>::Scheduler;
};

template <class ArchTag>
struct TileSchedulerSelector<
    IndividualScheduler,
    ArchTag,
    cute::enable_if_t<cute::is_same_v<ArchTag, cutlass::arch::IntelXe>>> {
  using Scheduler = kernel::XeFlashIndividualTileScheduler;
};
} // namespace detail

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::flash_attention
