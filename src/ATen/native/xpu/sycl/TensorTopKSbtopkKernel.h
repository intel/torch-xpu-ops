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

// Try to run topk using the sbtopk (single-block topk) kernel.
// Returns true if sbtopk was used, false if the caller should fall back
// to the original group radix select.
//
// This function is compiled in a separate translation unit
// (TensorTopKSbtopkKernel.cpp) to isolate sbtopk's template instantiations
// from the original topk kernel. The SYCL compiler's global optimization
// decisions are affected by the total set of templates in a compilation unit;
// keeping sbtopk separate prevents it from regressing the original kernel's
// performance on small-dim cases where sbtopk is not even used.
//
// Note: this function only performs the radix selection (unsorted).
// The caller is responsible for sorting the k results if needed, since
// segmented_sort_pairs lives in SortingKernels.h and including it here
// would defeat the purpose of compilation unit isolation.
TORCH_XPU_API bool sbtopk_try_launch(
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
