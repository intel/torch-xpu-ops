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

#include <c10/util/Exception.h>
#include <comm/DeviceProperties.h>
#include <algorithm>
#include <limits>

#define XPU_KERNEL_LOOP_TYPE(item, i, n, index_type)                      \
  int64_t _i_n_d_e_x =                                                    \
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0); \
  for (index_type i = _i_n_d_e_x; _i_n_d_e_x < (n);                       \
       _i_n_d_e_x += item.get_local_range(0) * item.get_group_range(0),   \
                  i = _i_n_d_e_x)

#define XPU_KERNEL_LOOP(item, i, n) XPU_KERNEL_LOOP_TYPE(item, i, n, int)

constexpr int SYCL_NUM_THREADS = 1024;

inline int GET_GROUPS(
    const int64_t N,
    const int64_t max_threads_per_group = SYCL_NUM_THREADS) {
  TORCH_INTERNAL_ASSERT(
      N > 0, "XPU kernel launch blocks must be positive, but got N=", N);
  constexpr int64_t max_int = std::numeric_limits<int>::max();

  // Round up division for positive number that cannot cause integer overflow
  auto group_num = (N - 1) / max_threads_per_group + 1;
  TORCH_INTERNAL_ASSERT(
      group_num <= max_int, "Can't schedule too many blocks on XPU device");

  return static_cast<int>(group_num);
}

// Grid-strided loop kernels (see XPU_KERNEL_LOOP) must not launch one work item
// per element; doing so degenerates the strided loop into a no-op. Cap the
// number of launched work items at roughly the number the device can keep
// resident so each work item processes multiple elements. The 32 factor is the
// max sub-group (SIMD) width on Intel GPUs, so this estimates
// EU count * HW threads per EU * SIMD lanes.
inline int64_t syclMaxWorkItemsForLoop(
    at::DeviceIndex dev_id = at::xpu::current_device()) {
  return xpu::sycl::syclGpuEuCount(dev_id) *
      xpu::sycl::syclGpuHWThreadsPerEU(dev_id) * 32;
}

// Returns the number of work groups (not work items) needed to cover `nelem`
// elements with the given work-group size, capped by syclMaxWorkItemsForLoop().
inline int64_t syclLoopGroupRange(
    int64_t nelem,
    int64_t group_size = SYCL_NUM_THREADS) {
  int64_t work_items = std::min(nelem, syclMaxWorkItemsForLoop());
  return (work_items + group_size - 1) / group_size;
}
