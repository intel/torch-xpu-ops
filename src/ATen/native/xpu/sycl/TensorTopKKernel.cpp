/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// ============================================================================
// torch.topk XPU kernel implementation
//
// This file provides two topk selection strategies:
//
// 1. Original group radix select (segmented_group_select_pairs):
//    Tournament-style radix select from SortingKernels.h. Uses
//    RADIX_BITS=4, KEYS_PER_ITEM=4, GROUP_SIZE=1024 to process 4096
//    elements per work-group per round. Well-suited for small batch sizes
//    or very small slice dimensions. Supports k up to 256.
//
// 2. Single-block topk (sbtopk) — in separate compilation unit:
//    Compiled in TensorTopKSbtopkKernel.cpp as a separate .so to avoid
//    SYCL compiler global optimization interference with this file.
//    Uses RADIX_BITS=8, VEC=4 vectorized loads, BLOCK_SIZE=256.
//    One work-group per slice. See TensorTopKSbtopkKernel.cpp for details.
//
// Dispatch logic (topk_kernel):
//   - k > 256           → full sort fallback (topk_out_with_sort)
//   - sbtopk eligible   → sbtopk (k <= 16, nelements > 4096, enough slices
//                          to saturate GPU occupancy)
//   - otherwise          → original group radix select
// ============================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Sorting.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>
#include <ATen/native/xpu/sycl/SortingRadixSelect.h>

#include <ATen/native/xpu/sycl/TensorTopKKernel.h>
#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernel.h>

namespace at {
namespace native {
namespace xpu {

// Fallback for k > 256: perform a full sort and take the first k elements.
// This is simpler but O(n log n) instead of O(n), so only used when k is
// too large for the segmented radix select (which supports k <= 256).
void topk_out_with_sort(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    const Tensor& values,
    const Tensor& indices) {
  Tensor sorted_values, sorted_indices;
  std::tie(sorted_values, sorted_indices) =
      at::sort(self, /* stable= */ false, dim, largest);
  values.copy_(sorted_values.narrow(dim, 0, k));
  indices.copy_(sorted_indices.narrow(dim, 0, k));
}

// Main topk dispatch function.
//
// Dispatch strategy (in priority order):
//   1. k > 256  → topk_out_with_sort (full sort, then slice)
//   2. sbtopk   → single-block radix select (k <= 16, large slices,
//                  enough slices to saturate GPU occupancy)
//                  [compiled in separate TensorTopKSbtopkKernel.cpp]
//   3. default  → segmented_group_select_pairs (original tournament
//                  radix select from SortingKernels.h, k <= 256)
//
// After selection, if sorted=true, the k results are sorted using
// segmented_sort_pairs (a radix sort on k elements per slice).
void topk_kernel(
    const at::Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    const at::Tensor& values,
    const at::Tensor& indices) {
  if (k == 0) {
    return;
  }

  TORCH_CHECK(
      input.defined() && values.defined() && indices.defined(),
      "invalid inputs");

  auto self = (input.dim() == 0) ? input.view(1) : input;

  int64_t numel = self.numel();
  int64_t ndim = self.dim();
  dim = maybe_wrap_dim(dim, ndim);
  int64_t nelements = self.sizes()[dim];  // slice size (= dimension being selected)
  int64_t nsegments = numel / nelements;  // number of independent topk problems

  TORCH_CHECK(
      nelements <= std::numeric_limits<int>::max(),
      "The dimension being select can not have more than INT_MAX elements.");

  const auto self_dtype = self.dtype();
  TORCH_CHECK(
      self_dtype != ScalarType::ComplexFloat &&
          self_dtype != ScalarType::ComplexDouble,
      "Topk currently does not support complex dtypes on XPU.");

  auto out_sizes = self.sizes().vec();
  out_sizes[dim] = k;
  values.resize_(out_sizes);
  indices.resize_(out_sizes);

  if (k > 256) { // The segmented_group_select_pairs supports k<=256
    topk_out_with_sort(self.contiguous(), k, dim, largest, values, indices);
    return;
  }

  // Both sbtopk and the original radix select operate on the last dimension.
  // If dim != last, transpose so the target dimension becomes contiguous.
  Tensor self_;
  bool need_infer_dim = dim != ndim - 1;
  if (!need_infer_dim) {
    self_ = self.contiguous();
  } else {
    self_ = self.transpose(ndim - 1, dim).contiguous();
    std::swap(out_sizes[ndim - 1], out_sizes[dim]);
  }

  // Allocate output tensors (or reuse if already contiguous on last dim).
  Tensor values_, indices_;
  bool newvalues = false;
  bool newindices = false;
  if (!need_infer_dim && values.is_contiguous()) {
    values_ = values;
  } else {
    values_ = at::empty(out_sizes, values.options());
    newvalues = true;
  }
  if (!need_infer_dim && indices.is_contiguous()) {
    indices_ = indices;
  } else {
    indices_ = at::empty(out_sizes, indices.options());
    newindices = true;
  }

  // Try sbtopk first (compiled in separate TensorTopKSbtopkKernel.cpp).
  // sbtopk_try_launch only does radix selection (unsorted); sorting is
  // handled below together with the original path.
  bool used_sbtopk = sbtopk_try_launch(
      self_, nsegments, nelements, k, largest, values_, indices_);

  if (!used_sbtopk) {
    // Fall back to original tournament radix select.
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self_.scalar_type(),
        "topk_xpu",
        [&]() {
          scalar_t* self_ptr = self_.data_ptr<scalar_t>();
          scalar_t* values_ptr = values_.data_ptr<scalar_t>();
          int64_t* indices_ptr = indices_.data_ptr<int64_t>();

          // Original tournament radix select: RADIX_BITS=4, KPI=4,
          // GROUP_SIZE=1024, processes 4096 elements per round.
          segmented_group_select_pairs<scalar_t, int64_t>(
              self_ptr,
              (scalar_t*)values_ptr,
              nullptr,
              (int64_t*)indices_ptr,
              nsegments,
              nelements,
              k,
              largest);
        });
  }

  // Both sbtopk and radix select output unsorted top-k results.
  // If sorted=true, run a radix sort on the k result elements per slice.
  if (sorted) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self_.scalar_type(),
        "topk_sort_xpu",
        [&]() {
          scalar_t* values_ptr = values_.data_ptr<scalar_t>();
          int64_t* indices_ptr = indices_.data_ptr<int64_t>();
          segmented_sort_pairs<scalar_t, int64_t>(
              values_ptr,
              values_ptr,
              indices_ptr,
              indices_ptr,
              nsegments,
              k,
              largest);
        });
  }

  // If we transposed earlier, transpose results back to the original layout
  // and copy into the caller's output tensors.
  if (newvalues) {
    if (need_infer_dim)
      values_.transpose_(ndim - 1, dim);
    values.copy_(values_);
  }
  if (newindices) {
    if (need_infer_dim)
      indices_.transpose_(ndim - 1, dim);
    indices.copy_(indices_);
  }
}

} // namespace xpu
} // namespace native
} // namespace at
