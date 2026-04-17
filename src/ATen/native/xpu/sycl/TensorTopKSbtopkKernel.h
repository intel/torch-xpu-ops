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
//   FAILED   - did not run; caller should fall back to original kernel.
//   UNSORTED - ran; output contains top-k values but is not sorted.
//              Caller must sort if sorted output is requested.
//   SORTED   - ran; output is already sorted (descending for largest,
//              ascending for smallest). Caller can skip sort.
enum class SbtopkResult : int {
  FAILED   = 0,
  UNSORTED = 1,
  SORTED   = 2,
};

// Try to run topk using the subgroup topk kernel (separate TU to avoid
// SYCL compiler interference with the original kernel's codegen).
TORCH_XPU_API SbtopkResult sbtopk_try_launch(
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
