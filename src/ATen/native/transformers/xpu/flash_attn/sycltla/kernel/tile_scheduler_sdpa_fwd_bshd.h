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
