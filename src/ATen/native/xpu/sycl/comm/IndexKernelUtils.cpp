/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/sycl/comm/IndexKernelUtils.h>

namespace at::native::xpu {

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
    bool allow_neg_indices) {
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

} // namespace at::native::xpu
