/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
    // FastDivmod divmod_num_heads;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  XeFlashIndividualTileScheduler(Params const& params) : params(params) {}

  template <class ProblemSize, class TileShape>
  static Params to_underlying_arguments(
      ProblemSize const& problem_size,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape) {
    using namespace cute;
    dim3 grid(
        size(ceil_div(
            shape<3>(problem_size),
            shape<0>(tile_shape))), // seq_len_qo / 128
        size(shape<1>(problem_size)), // num_heads_q
        size(shape<0>(problem_size))); // batch
    return Params{grid};
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
    return make_coord(BlockIdxX(), BlockIdxY(), BlockIdxZ());
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
