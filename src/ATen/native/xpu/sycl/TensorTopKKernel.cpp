/*
 * Copyright 2020-2025 Intel Corporation
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
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Sorting.h>
#include <ATen/native/xpu/sycl/SortingKernels.h>

#include <ATen/native/xpu/sycl/TensorTopKKernel.h>

namespace at {
namespace native {
namespace xpu {

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

  if (k > 256) { // The segmented_group_select_pairs supports k<=256
    topk_out_with_sort(self.contiguous(), k, dim, largest, values, indices);
    return;
  }

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

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self_.scalar_type(),
      "topk_xpu",
      [&]() {
        scalar_t* self_ptr = self_.data_ptr<scalar_t>();
        scalar_t* values_ptr = values_.data_ptr<scalar_t>();
        int64_t* indices_ptr = indices_.data_ptr<int64_t>();
        segmented_group_select_pairs<scalar_t, int64_t>(
            self_ptr,
            (scalar_t*)values_ptr,
            nullptr,
            (int64_t*)indices_ptr,
            nsegments,
            nelements,
            k,
            largest);

        if (sorted) {
          segmented_sort_pairs<scalar_t, int64_t>(
              values_ptr,
              values_ptr,
              indices_ptr,
              indices_ptr,
              nsegments,
              k,
              largest);
        }
      });

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
