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
#include <cutlass/fast_math.h>
#include <cutlass/kernel_hardware_info.h>

namespace cutlass::fmha::kernel {

struct XeFMHAIndividualTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFMHAIndividualTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo,
      TileShape const& tile_shape) {
    using namespace cute;

    dim3 grid(
        size(ceil_div(shape.head_size_vo, get<1>(tile_shape))), // V
        size(ceil_div(shape.seq_len_qo, get<0>(tile_shape))), // Q
        size(shape.batch * shape.num_heads_q)); // (h,b) -- split later
    return Params{grid, {shape.num_heads_q}};
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
  auto get_block_coord() {
    using namespace cute;
    int idx_b = BlockIdxZ();
    int head;
    params.divmod_num_heads(idx_b, head, idx_b);
    return make_coord(BlockIdxY(), BlockIdxX(), head, idx_b);
  }

  CUTLASS_DEVICE
  XeFMHAIndividualTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

} // namespace cutlass::fmha::kernel
