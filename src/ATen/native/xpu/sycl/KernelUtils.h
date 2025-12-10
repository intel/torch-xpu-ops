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

#include <c10/util/Exception.h>
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
