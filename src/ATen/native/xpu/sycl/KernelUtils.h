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
#include <algorithm>

#define XPU_KERNEL_LOOP_TYPE(item, i, n, index_type)                      \
  int64_t _i_n_d_e_x =                                                    \
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0); \
  for (index_type i = _i_n_d_e_x; _i_n_d_e_x < (n);                       \
       _i_n_d_e_x += item.get_local_range(0) * item.get_group_range(0),   \
                  i = _i_n_d_e_x)

#define XPU_KERNEL_LOOP(item, i, n) XPU_KERNEL_LOOP_TYPE(item, i, n, int)

// Returns the number of work groups needed to cover `nelem` elements with the
// given work-group size, capped so grid-strided loop kernels (XPU_KERNEL_LOOP)
// do not launch more work items than the device can keep resident. The default
// cap is syclMaxWorkItemsPerTile(); pass a custom `candidate` to override.
inline int64_t xpuKernelLoopGroupRange(
    int64_t nelem,
    int64_t group_size = xpu::sycl::syclDeviceMaxWorkGroupSize(),
    int64_t candidate = xpu::sycl::syclMaxWorkItemsPerTile()) {
  int64_t work_items = std::min(nelem, candidate);
  return at::ceil_div(work_items, group_size);
}
