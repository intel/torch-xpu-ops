/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/xpu/sycl/TensorTopKRadixSelect.h>

#include <ATen/native/xpu/sycl/TensorTopKKernel.h>

namespace at {
namespace native {
namespace xpu {

// TopK kernel for XPU: uses radix-select instead of full sort.
// The radix_topk_kernel produces unsorted top-K results; if the caller
// requests sorted output, we sort the K results post-hoc (cheap since K
// is typically small).
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
  int64_t nelements = self.sizes()[dim];
  int64_t nsegments = numel / nelements;

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

  // For non-last-dim topk, transpose so the target dim is last (contiguous),
  // run radix select on the last dim, then transpose back.
  Tensor self_;
  bool need_infer_dim = dim != ndim - 1;
  if (!need_infer_dim) {
    self_ = self.contiguous();
  } else {
    self_ = self.transpose(ndim - 1, dim).contiguous();
    std::swap(out_sizes[ndim - 1], out_sizes[dim]);
  }

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

  auto& q = c10::xpu::getCurrentXPUStream().queue();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self_.scalar_type(),
      "topk_xpu",
      [&]() {
        const scalar_t* self_ptr = self_.const_data_ptr<scalar_t>();
        scalar_t* values_ptr = values_.data_ptr<scalar_t>();
        int64_t* indices_ptr = indices_.data_ptr<int64_t>();

        radix_topk_kernel<scalar_t>(
            self_ptr,
            values_ptr,
            indices_ptr,
            nsegments,
            nelements,
            k,
            largest,
            q);
      });

  // Sort the K results if requested. Radix select produces unsorted output,
  // so we sort the (small) K-element slices and rearrange indices to match.
  if (sorted && k > 1) {
    auto sorted_result = values_.sort(-1, largest);
    at::Tensor sort_perm = std::get<1>(sorted_result);
    values_.copy_(std::get<0>(sorted_result));
    indices_.copy_(indices_.gather(-1, sort_perm));
  }

  // Copy results back if we created temporary tensors (non-last-dim case)
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
