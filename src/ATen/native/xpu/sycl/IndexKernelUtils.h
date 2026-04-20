/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ceil_div.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <comm/SYCLContext.h>
#include <cstdint>
namespace at::native::xpu {

template <int alignment>
inline bool fast_gather_kernel_eligible(
    const TensorIterator& iter,
    char* const out_ptr,
    char* const in_ptr,
    const size_t index_stride_bytes,
    const size_t element_size) {
  using at::native::memory::get_alignment;
  const auto index_element_size = iter.element_size(2);
  // TensorIterator strides and sizes are ordered fastest moving to slowest
  // moving, in contrast to regular sizes
  // we need contiguous source and dst slices and aligned pointers and strides
  // and slice size to do vectorized loads also we need idx to be expanded in
  // the last dimension so we can copy entire slices and we need the src tensor
  // to keep 0 stride from restriding (it could have been deleted by dimension
  // collapse, in this case iterator would still be 2d but we cannot use fast
  // path)

  return iter.ndim() == 2 && iter.strides(2)[0] == 0 &&
      iter.strides(2)[1] == index_element_size &&
      static_cast<size_t>(iter.strides(0)[0]) == element_size &&
      static_cast<size_t>(iter.strides(1)[0]) == element_size &&
      static_cast<size_t>(iter.strides(1)[1] == 0) &&
      get_alignment(out_ptr) == alignment &&
      get_alignment(in_ptr) == alignment &&
      get_alignment(static_cast<size_t>(iter.shape()[0] * element_size)) ==
      alignment &&
      get_alignment(static_cast<size_t>(index_stride_bytes)) == alignment &&
      get_alignment(static_cast<size_t>(iter.strides(0)[1])) == alignment;
}

// Eligibility check for vectorized scatter (is_scatter_like=true).
//
// For scatter with TensorAssign:
//   dst[index[i], j] = src[i, j]
// TensorIterator layout (fastest-to-slowest):
//   strides(0) = dst: [element_size, 0]  ← dim restrided to 0
//   strides(1) = src: [element_size, dim_size*element_size]  ← original
//   strides(2) = idx: [0, idx_elem_size]  ← expanded along dim
template <int alignment>
inline bool fast_scatter_kernel_eligible(
    const TensorIterator& iter,
    char* const dst_ptr,
    char* const src_ptr,
    const size_t index_stride_bytes,
    const size_t element_size) {
  using at::native::memory::get_alignment;
  const auto index_element_size = iter.element_size(2);

  return iter.ndim() == 2 && iter.strides(2)[0] == 0 &&
      iter.strides(2)[1] == index_element_size &&
      static_cast<size_t>(iter.strides(0)[0]) == element_size &&
      static_cast<size_t>(iter.strides(1)[0]) == element_size &&
      static_cast<size_t>(iter.strides(0)[1]) == 0 &&
      get_alignment(dst_ptr) == alignment &&
      get_alignment(src_ptr) == alignment &&
      get_alignment(static_cast<size_t>(iter.shape()[0] * element_size)) ==
      alignment &&
      get_alignment(static_cast<size_t>(index_stride_bytes)) == alignment &&
      get_alignment(static_cast<size_t>(iter.strides(1)[1])) == alignment;
}

#define SIMD 32

template <int Alignment, typename index_t>
struct VectorizedGatherKernel {
  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<2> item) const {
    int64_t ind = idx_[item.get_group(1)];
    if (allow_neg_indices_) {
      ind = (ind < 0) ? ind + ind_dim_size_ : ind;
    }
    SYCL_KERNEL_ASSERT(
        ind >= 0 && ind < ind_dim_size_ &&
        "vectorized gather kernel index out of bounds");
    int32_t off =
        (item.get_local_range(1) * item.get_group(0) + item.get_local_id(1)) *
        Alignment; // off is guaranteed to be within int32 limits
    if (off >= slice_size_)
      return;
    auto vec =
        at::native::memory::ld_vec<Alignment>(inp_ + ind * inp_stride_ + off);
    at::native::memory::st_vec<Alignment>(
        out_ + item.get_group(1) * (int32_t)out_stride_ + off,
        vec); // out offset is guaranteed to be within int32 limits
  }
  VectorizedGatherKernel(
      char* out,
      char* inp,
      index_t* idx,
      int num_ind,
      int64_t slice_size,
      int64_t ind_dim_size,
      int64_t inp_stride,
      int64_t out_stride,
      bool allow_neg_indices)
      : out_(out),
        inp_(inp),
        idx_(idx),
        num_ind_(num_ind),
        slice_size_(slice_size),
        ind_dim_size_(ind_dim_size),
        inp_stride_(inp_stride),
        out_stride_(out_stride),
        allow_neg_indices_(allow_neg_indices) {}

 private:
  char* out_;
  char* inp_;
  index_t* idx_;
  int num_ind_;
  int64_t slice_size_;
  int64_t ind_dim_size_;
  int64_t inp_stride_;
  int64_t out_stride_;
  bool allow_neg_indices_;
};

template <int64_t Alignment, typename index_t>
void vectorized_gather_kernel_launch(
    char* out,
    char* inp,
    index_t* idx,
    int num_ind,
    int64_t slice_size_in_bytes,
    int64_t ind_dim_size,
    int64_t inp_stride_bytes,
    int64_t out_stride_bytes,
    bool allow_neg_indices = false) {
  int64_t max_num_threads = syclMaxWorkItemsPerSubSlice();
  auto num_threads = at::round_up(
      at::ceil_div(slice_size_in_bytes, Alignment), static_cast<int64_t>(SIMD));
  auto wg_size = std::min(max_num_threads, num_threads);
  sycl::range<2> local_range(1, wg_size);
  sycl::range<2> global_range(
      static_cast<uint32_t>(
          at::ceil_div(slice_size_in_bytes, max_num_threads * Alignment)),
      static_cast<uint32_t>(num_ind) * wg_size);
  auto caller = VectorizedGatherKernel<Alignment, index_t>(
      out,
      inp,
      idx,
      num_ind,
      slice_size_in_bytes,
      ind_dim_size,
      inp_stride_bytes,
      out_stride_bytes,
      allow_neg_indices);
  sycl_kernel_submit(
      global_range, local_range, at::xpu::getCurrentSYCLQueue(), caller);
}

// explicit template instantiation
template void vectorized_gather_kernel_launch<16, int64_t>(
    char* out,
    char* inp,
    int64_t* idx,
    int num_ind,
    int64_t slice_size_in_bytes,
    int64_t ind_dim_size,
    int64_t inp_stride_bytes,
    int64_t out_stride_bytes,
    bool allow_neg_indices);

// explicit template instantiation
template void vectorized_gather_kernel_launch<16, int32_t>(
    char* out,
    char* inp,
    int32_t* idx,
    int num_ind,
    int64_t slice_size_in_bytes,
    int64_t ind_dim_size,
    int64_t inp_stride_bytes,
    int64_t out_stride_bytes,
    bool allow_neg_indices);

// Vectorized scatter kernel: copies contiguous slices from src to indexed
// positions in dst using wide (Alignment-byte) loads and stores.
//
// For scatter_(dim, index, src):
//   For each index i in [0, num_ind):
//     dst[idx[i], :] = src[i, :]
//
// Each work-group handles one index entry and copies the entire slice
// using Alignment-byte vector load/store operations.
// This avoids narrow d16 stores for bf16/fp16 by packing multiple elements
// into wider d32/d64/d128 transactions, matching fp32 store throughput.
template <int Alignment, typename index_t>
struct VectorizedScatterKernel {
  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(sycl::nd_item<2> item) const {
    int64_t ind = idx_[item.get_group(1)];
    SYCL_KERNEL_ASSERT(
        ind >= 0 && ind < ind_dim_size_ &&
        "vectorized scatter kernel index out of bounds");
    int32_t off =
        (item.get_local_range(1) * item.get_group(0) + item.get_local_id(1)) *
        Alignment;
    if (off >= slice_size_)
      return;
    const int64_t inp_group_offset =
        static_cast<int64_t>(item.get_group(1)) * inp_stride_;
    auto vec =
        at::native::memory::ld_vec<Alignment>(inp_ + inp_group_offset + off);
    at::native::memory::st_vec<Alignment>(out_ + ind * out_stride_ + off, vec);
  }
  VectorizedScatterKernel(
      char* out,
      char* inp,
      index_t* idx,
      int num_ind,
      int64_t slice_size,
      int64_t ind_dim_size,
      int64_t inp_stride,
      int64_t out_stride)
      : out_(out),
        inp_(inp),
        idx_(idx),
        num_ind_(num_ind),
        slice_size_(slice_size),
        ind_dim_size_(ind_dim_size),
        inp_stride_(inp_stride),
        out_stride_(out_stride) {}

 private:
  char* out_;
  char* inp_;
  index_t* idx_;
  int num_ind_;
  int64_t slice_size_;
  int64_t ind_dim_size_;
  int64_t inp_stride_;
  int64_t out_stride_;
};

template <int64_t Alignment, typename index_t>
void vectorized_scatter_kernel_launch(
    char* out,
    char* inp,
    index_t* idx,
    int num_ind,
    int64_t slice_size_in_bytes,
    int64_t ind_dim_size,
    int64_t inp_stride_bytes,
    int64_t out_stride_bytes) {
  int64_t max_num_threads = syclMaxWorkItemsPerSubSlice();
  auto num_threads = at::round_up(
      at::ceil_div(slice_size_in_bytes, Alignment), static_cast<int64_t>(SIMD));
  auto wg_size = std::min(max_num_threads, num_threads);
  sycl::range<2> local_range(1, wg_size);
  sycl::range<2> global_range(
      static_cast<uint32_t>(
          at::ceil_div(slice_size_in_bytes, max_num_threads * Alignment)),
      static_cast<uint32_t>(num_ind) * wg_size);
  auto caller = VectorizedScatterKernel<Alignment, index_t>(
      out,
      inp,
      idx,
      num_ind,
      slice_size_in_bytes,
      ind_dim_size,
      inp_stride_bytes,
      out_stride_bytes);
  sycl_kernel_submit(
      global_range, local_range, at::xpu::getCurrentSYCLQueue(), caller);
}

} // namespace at::native::xpu
