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

#include <ATen/ceil_div.h>
#include <comm/DeviceProperties.h>

// Keep XPUContext after the existing host-only SYCL warning suppression.
#include <ATen/xpu/XPUContext.h>
#include <algorithm>
#include <limits>
#include <type_traits>

#define XPU_KERNEL_LOOP_TYPE(item, i, n, index_type)                      \
  int64_t _i_n_d_e_x =                                                    \
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0); \
  for (index_type i = _i_n_d_e_x; _i_n_d_e_x < (n);                       \
       _i_n_d_e_x += item.get_local_range(0) * item.get_group_range(0),   \
                  i = _i_n_d_e_x)

#define XPU_KERNEL_LOOP(item, i, n) XPU_KERNEL_LOOP_TYPE(item, i, n, int)

// Grid-stride range whose counter, bound and stride are all index_t. Unlike
// XPU_KERNEL_LOOP_TYPE, which always counts in int64_t, a 32-bit index_t keeps
// the loop in native 32-bit instructions (64-bit integer compares and
// multiplies are emulated on Intel GPUs). The caller must check
// xpuKernelLoopFitsIndex<index_t>(n, global_range) on the host so the final
// increment cannot overflow index_t.
template <typename index_t>
class XpuKernelLoopRange {
  static_assert(std::is_integral_v<index_t>);

 public:
  class iterator {
   public:
    iterator(index_t i, index_t stride) : i_(i), stride_(stride) {}
    index_t operator*() const {
      return i_;
    }
    iterator& operator++() {
      i_ += stride_;
      return *this;
    }
    bool operator!=(const iterator& end) const {
      return i_ < end.i_;
    }

   private:
    index_t i_;
    index_t stride_;
  };

  template <typename item_t>
  XpuKernelLoopRange(const item_t& item, index_t n)
      : begin_(static_cast<index_t>(item.get_global_id(0))),
        end_(n),
        stride_(static_cast<index_t>(item.get_global_range(0))) {}

  iterator begin() const {
    return {begin_, stride_};
  }
  iterator end() const {
    return {end_, stride_};
  }

 private:
  index_t begin_;
  index_t end_;
  index_t stride_;
};

// Grid-stride loop over [0, n) computed in index_type; `i` is const in the
// body. Requires xpuKernelLoopFitsIndex<index_type>(n, global_range).
#define XPU_KERNEL_LOOP_INDEX(item, i, n, index_type) \
  for (const index_type i : XpuKernelLoopRange<index_type>(item, n))

// Whether XPU_KERNEL_LOOP_INDEX over [0, n) launched with `global_range` work
// items can use index_t: every start index and the last increment past n must
// stay within index_t.
template <typename index_t>
inline bool xpuKernelLoopFitsIndex(int64_t n, int64_t global_range) {
  return n >= 0 && global_range >= 0 &&
      n <=
      static_cast<int64_t>(std::numeric_limits<index_t>::max()) - global_range;
}

// Returns the number of work groups needed to cover `nelem` elements with the
// given work-group size, capped so grid-strided loop kernels (XPU_KERNEL_LOOP)
// do not launch more work items than the device can keep resident. The default
// cap is at::xpu::getDeviceMaxWorkItems(); pass a custom `candidate` to
// override.
inline int64_t xpuKernelLoopGroupRange(
    int64_t nelem,
    int64_t group_size = at::xpu::getDeviceMaxWorkGroupSize(),
    int64_t candidate = at::xpu::getDeviceMaxWorkItems()) {
  int64_t work_items = std::min(nelem, candidate);
  return at::ceil_div(work_items, group_size);
}
