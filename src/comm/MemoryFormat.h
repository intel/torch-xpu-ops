#pragma once

#include <ATen/core/Tensor.h>
#include <comm/SYCLContext.h>

// Channels last 1D is only supported by IPEX GPU
#define CL1D_OFFSET 20
#define CHANNELSLAST1D_SYCL \
  ((at::MemoryFormat)((int)at::MemoryFormat::ChannelsLast3d + CL1D_OFFSET))

using namespace at;

namespace xpu {
namespace sycl {

inline std::vector<int64_t> get_channels_last_strides_1d_sycl(
    IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  switch (sizes.size()) {
    case 3:
      strides[1] = 1;
      strides[2] = sizes[1];
      strides[0] = strides[2] * sizes[2];
      return strides;
    case 2:
      strides[0] = 1;
      strides[1] = sizes[0];
      return strides;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "ChannelsLast1d doesn't support size ", sizes.size());
  }
}

inline bool is_channels_last_strides_1d_s3_sycl(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  int64_t min = 0;
  // special case for trivial C dimension. default to NCL
  if (strides[1] == 0) {
    return false;
  }
  // loop strides indices
  for (auto& d : {1, 2, 0}) {
    if (sizes[d] == 0) {
      return false;
    }
    if (strides[d] < min) {
      return false;
    }
    // Fallback to NCL as default layout for ambiguous cases
    // This is the flaw of implicit memory_format from strides.
    // N11 tensor with identical strides for size 1 dimension;
    // Two cases could lead us here:
    // a. N11 contiguous Tensor ([N,1,1]@[1,1,1])
    // b. N1L contiguous Tensor sliced on the L-dimension. ([N,1,1]@[L,L,L])
    if (d == 0 && min == strides[1]) {
      return false;
    }

    min = strides[d];
    if (sizes[d] > 1) {
      min *= sizes[d];
    }
  }
  return true;
}

inline bool is_channels_last_strides_1d_sycl(
    const IntArrayRef sizes,
    const IntArrayRef strides) {
  switch (sizes.size()) {
    case 3:
      return is_channels_last_strides_1d_s3_sycl(sizes, strides);
    case 2:
      // TODO dim == 2 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

inline bool compute_strides_like_channels_last_1d_sycl(const Tensor& t) {
  return is_channels_last_strides_1d_sycl(t.sizes(), t.strides());
}

inline at::MemoryFormat suggest_memory_format_sycl(
    const Tensor& t,
    bool channels_last_strides_exact_match = false) {
  const auto ndim = t.ndimension();
  if (3 != ndim) {
    return t.suggest_memory_format();
  }

  // ipex gpu channels last 1d
  if (!t.is_mkldnn() && !t.is_sparse()) {
    if (compute_strides_like_channels_last_1d_sycl(t)) {
      if (!channels_last_strides_exact_match ||
          get_channels_last_strides_1d_sycl(t.sizes()) == t.strides()) {
        return CHANNELSLAST1D_SYCL;
      }
    }
  }

  return at::MemoryFormat::Contiguous;
}

inline bool is_channels_last(at::MemoryFormat fmt) {
  if (
#ifdef USE_CHANNELS_LAST_1D
      CHANNELSLAST1D_SYCL == fmt ||
#endif
      at::MemoryFormat::ChannelsLast == fmt ||
      at::MemoryFormat::ChannelsLast3d == fmt) {
    return true;
  }
  return false;
}

inline bool is_smf_channels_last(const Tensor& t) {
  const auto ndim = t.ndimension();
  if (
#ifdef USE_CHANNELS_LAST_1D
      3 != ndim &&
#endif
      4 != ndim && 5 != ndim) {
    // channels last only supports 3D, 4D, 5D tensor
    return false;
  }

  return is_channels_last(suggest_memory_format_sycl(t));
}

} // namespace sycl
} // namespace xpu
