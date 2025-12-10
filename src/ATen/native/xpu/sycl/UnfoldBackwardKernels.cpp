/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnfoldBackward.h>
#include <c10/core/ScalarType.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/UnfoldBackwardKernels.h>

namespace at::native::xpu {

constexpr int n_elems_per_work_item = 4; // UNROLLED_ELEM_PER_WORK_ITEM;

template <int n_elems_per_work_item, typename func_t>
struct UnfoldBackwardElementwiseKernelFunctor {
  void operator()(sycl::item<1> item) const {
    int idx = item.get_linear_id();
#pragma unroll
    for (int i = 0; i < n_elems_per_work_item; ++i) {
      if (idx < total_n_elems_) {
        f_(idx);
        idx += total_work_items_;
      }
    }
  }
  UnfoldBackwardElementwiseKernelFunctor(
      int total_work_items,
      int total_n_elems,
      func_t f)
      : total_work_items_(total_work_items),
        total_n_elems_(total_n_elems),
        f_(f) {}

 private:
  int total_work_items_;
  int total_n_elems_;
  func_t f_;
};

template <int n_elems_per_work_item, typename func_t>
static void _launch_unfold_backward_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <=
          std::numeric_limits<int32_t>::max()); // INT_MAX when int32_t

  int total_work_items =
      (total_n_elems + n_elems_per_work_item - 1) / n_elems_per_work_item;
  UnfoldBackwardElementwiseKernelFunctor<n_elems_per_work_item, func_t> kfn(
      total_work_items, total_n_elems, f);
  auto& queue = getCurrentSYCLQueue();

  sycl_kernel_submit(sycl::range<1>(total_work_items), queue, kfn);
}

template <typename scalar_t, typename offset_calc_t>
struct UnfoldBackwardFunctor {
  void operator()(int i) const {
    auto offsets = offset_calc_.get(i);

    auto* grad_out_data =
        reinterpret_cast<scalar_t*>(grad_out_ptr_ + offsets[0]);
    auto* grad_in_data = reinterpret_cast<scalar_t*>(grad_in_ptr_ + offsets[1]);

    auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr_ + offsets[2]);

    // left_fold potentially intersecting with idx_dim
    // is either (idx_dim - size) / step or the next integer.
    int64_t left_fold_idx = (idx_dim > size_) ? (idx_dim - size_) / step_ : 0;
    if (!(left_fold_idx * step_ <= idx_dim &&
          idx_dim < left_fold_idx * step_ + size_)) {
      ++left_fold_idx;
    }

    auto right_fold_idx = idx_dim / step_;
    right_fold_idx = (right_fold_idx >= grad_in_dim_size_)
        ? (grad_in_dim_size_ - 1)
        : right_fold_idx;

    for (auto fold_idx = left_fold_idx; fold_idx <= right_fold_idx;
         ++fold_idx) {
      auto idx_last_dim = idx_dim - fold_idx * step_;
      *grad_out_data += grad_in_data
          [fold_idx * grad_in_dim_stride_ +
           idx_last_dim * grad_in_last_dim_stride_];
    }
  }

  UnfoldBackwardFunctor(
      char* grad_out_ptr,
      char* grad_in_ptr,
      char* idx_dim_ptr,
      offset_calc_t offset_calc,
      int64_t size,
      int64_t step,
      int64_t grad_in_dim_stride,
      int64_t grad_in_last_dim_stride,
      int64_t grad_in_dim_size)
      : grad_out_ptr_(grad_out_ptr),
        grad_in_ptr_(grad_in_ptr),
        idx_dim_ptr_(idx_dim_ptr),
        offset_calc_(offset_calc),
        size_(size),
        step_(step),
        grad_in_dim_stride_(grad_in_dim_stride),
        grad_in_last_dim_stride_(grad_in_last_dim_stride),
        grad_in_dim_size_(grad_in_dim_size) {}

 private:
  char* grad_out_ptr_;
  char* grad_in_ptr_;
  char* idx_dim_ptr_;
  offset_calc_t offset_calc_;
  int64_t size_;
  int64_t step_;
  int64_t grad_in_dim_stride_;
  int64_t grad_in_last_dim_stride_;
  int64_t grad_in_dim_size_;
};

template <typename scalar_t>
void _unfold_backward_internal_kernel(
    TensorIterator& iter,
    int64_t size,
    int64_t step,
    int64_t grad_in_dim_stride,
    int64_t grad_in_last_dim_stride,
    int64_t grad_in_dim_size) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unfold_backward_internal_kernel<scalar_t>(
          sub_iter,
          size,
          step,
          grad_in_dim_stride,
          grad_in_last_dim_stride,
          grad_in_dim_size);
    }
    return;
  }

  char* grad_out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* grad_in_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* idx_dim_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  auto offset_calc = make_offset_calculator<3>(iter);

  // The algorithm is: for each index in grad_out find
  // the elements contributing to it and sum them up.
  // Note: the algorithm does not require any synchronization.
  UnfoldBackwardFunctor<scalar_t, decltype(offset_calc)> loop(
      grad_out_ptr,
      grad_in_ptr,
      idx_dim_ptr,
      offset_calc,
      size,
      step,
      grad_in_dim_stride,
      grad_in_last_dim_stride,
      grad_in_dim_size);

  _launch_unfold_backward_kernel<n_elems_per_work_item>(iter.numel(), loop);
}

void unfold_backward_kernel(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds
  auto last_dim = maybe_wrap_dim(-1, grad_in.dim());

  auto grad_in_dim_stride = ensure_nonempty_stride(grad_in, dim);
  auto grad_in_last_dim_stride = ensure_nonempty_stride(grad_in, last_dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);

  TensorIterator iter = _make_unfold_backward_iter_over_grad_out(
      grad_out, grad_in, dim, size, step);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "unfold_backward_xpu",
      [&] {
        _unfold_backward_internal_kernel<scalar_t>(
            iter,
            size,
            step,
            grad_in_dim_stride,
            grad_in_last_dim_stride,
            grad_in_dim_size);
      });
}
} // namespace at::native::xpu
