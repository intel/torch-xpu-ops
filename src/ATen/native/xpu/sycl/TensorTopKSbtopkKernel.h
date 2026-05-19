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

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace xpu {

// Result of sbtopk_try_launch.
//   FAILED   - sbtopk did not run; caller should fall back to original.
//   UNSORTED - sbtopk ran; output contains top-k values but is not sorted.
//              Caller must sort if sorted output is requested.
//   SORTED   - sbtopk ran; output is already sorted (descending for largest,
//              ascending for smallest). Caller can skip sort.
enum class SbtopkResult : int {
  FAILED = 0,
  UNSORTED = 1,
  SORTED = 2,
};

// Try to run topk using an optimized kernel path.
//
// Dispatches between the subgroup topk kernel (sub-group bitonic merge,
// output SORTED) and the single workgroup topk kernel (radix select,
// output UNSORTED) based on (nsegments, nelements, k).
SbtopkResult sbtopk_try_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices);

} // namespace xpu
} // namespace native
} // namespace at
