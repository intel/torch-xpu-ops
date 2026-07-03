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

#include <cstddef>
#include <numeric>

namespace at::native::xpu::detail {

inline size_t align_workgroup_size_to_subgroup_multiple(
    size_t workgroup_size,
    size_t subgroup_size,
    size_t max_workgroup_size) {
  if (workgroup_size % subgroup_size == 0) {
    return workgroup_size;
  }

  size_t num_subgroups = (workgroup_size + subgroup_size - 1) / subgroup_size;
  size_t aligned_size = num_subgroups * subgroup_size;

  if (aligned_size > max_workgroup_size) {
    num_subgroups = max_workgroup_size / subgroup_size;
    if (num_subgroups == 0) {
      num_subgroups = 1;
    }
    aligned_size = num_subgroups * subgroup_size;
  }

  return aligned_size;
}

inline size_t align_workgroup_2d_to_subgroup_multiple(
    size_t wg_range_x,
    size_t wg_range_y,
    size_t subgroup_size,
    size_t max_workgroup_size) {
  size_t total = wg_range_x * wg_range_y;

  if (total % subgroup_size == 0) {
    return wg_range_y;
  }

  size_t g = std::gcd(wg_range_x, subgroup_size);
  size_t y_multiplier = subgroup_size / g;

  size_t new_wg_range_y =
      ((wg_range_y + y_multiplier - 1) / y_multiplier) * y_multiplier;

  if (wg_range_x * new_wg_range_y > max_workgroup_size) {
    new_wg_range_y =
        (max_workgroup_size / wg_range_x / y_multiplier) * y_multiplier;
    if (new_wg_range_y == 0) {
      new_wg_range_y = y_multiplier;
    }
  }

  return new_wg_range_y;
}

} // namespace at::native::xpu::detail
