/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from FBGEMM
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/native/xpu/sycl/ReorderBatchedAd.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/dispatch_macros.h>

#include <sycl/sycl.hpp>
#include <comm/SYCLContext.h>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace at {

template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

namespace native::xpu {

static uint32_t xpu_calc_xblock_count_base(int num_items, int threads_per_block) {
  TORCH_CHECK(
      threads_per_block <= syclDeviceMaxWorkGroupSize(),
      "Number of threads must be <=1024!");
  constexpr uint64_t max_blocks = 2147483647;
  const auto u_num_items = static_cast<uint64_t>(num_items);
  const auto u_threads = static_cast<uint64_t>(threads_per_block);
  const uint64_t blocks =
      u_num_items / u_threads + (u_num_items % u_threads != 0);
  return static_cast<uint32_t>(std::min(blocks, max_blocks));
}

static uint32_t xpu_calc_xblock_count(int num_items, int threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  return xpu_calc_xblock_count_base(num_items, threads_per_block);
}

template <typename Dtype>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void reorder_batched_ad_lengths_kernel_(
    const GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        cat_ad_lengths,
    const GenericPackedTensorAccessor<
        int32_t,
        1,
        at::RestrictPtrTraits,
        int32_t> batch_offsets,
    GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths) {
  const int32_t B = batch_offsets.size(0) - 1;

  const int32_t num_ads_in_batch = batch_offsets[B];
  auto item = syclext::this_work_item::get_nd_item<2>();
  const auto b_t =
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const int32_t num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const int32_t input_segment_start =
      broadcast_lengths ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets[b];

  for (auto i = item.get_local_id(1); i < num_ads_b;
       i += item.get_local_range(1)) {
    reordered_cat_ad_lengths[output_segment_start + i] = broadcast_lengths
        ? cat_ad_lengths[input_segment_start]
        : cat_ad_lengths[input_segment_start + i];
  }
}

void reorder_batched_ad_lengths_xpu_kernel(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    Tensor& reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths,
    const int32_t grid_size) {
  FBGEMM_DISPATCH_ALL_TYPES(
      cat_ad_lengths.scalar_type(),
      "reorder_batched_ad_lengths_xpu_kernel",
      [&] {
        sycl_kernel_submit<reorder_batched_ad_lengths_kernel_<scalar_t>>(
            sycl::range<2>(32 * grid_size, 32),
            sycl::range<2>(32, 32),
            getCurrentSYCLQueue(),
            0,
            cat_ad_lengths
                .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
            batch_offsets
                .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
            reordered_cat_ad_lengths
                .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
            T,
            broadcast_lengths);
      });
}

template <typename Dtype, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void narrow_broadcast_indices_kernel_(
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> cat_ad_offsets,
    const GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        cat_ad_indices,
    GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        reordered_cat_ad_indices,
    const int num_ads_in_batch,
    const int reordered_cat_ad_batches,
    const int subGroupSize) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto lane_id = item.get_local_id(0) % subGroupSize;
  const auto warp_id =
      (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
      subGroupSize;
  const auto table_idx = warp_id / num_ads_in_batch;
  const auto ads_idx = warp_id % num_ads_in_batch;
  const auto start_offset = cat_ad_offsets[table_idx];
  const auto end_offset = cat_ad_offsets[table_idx + 1];
  const auto num_ads = end_offset - start_offset;
  if (warp_id < reordered_cat_ad_batches) {
    for (auto i = lane_id; i < num_ads; i += subGroupSize) {
      reordered_cat_ad_indices
          [start_offset * num_ads_in_batch + ads_idx * num_ads + i] =
              cat_ad_indices[start_offset + i];
    }
  }
}

template <typename Dtype, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void narrow_batched_broadcast_indices_kernel_(
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> cat_ad_offsets,
    const GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        cat_ad_indices,
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> reordered_cat_ad_offsets,
    GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        reordered_cat_ad_indices,
    const GenericPackedTensorAccessor<
        int32_t,
        1,
        at::RestrictPtrTraits,
        int32_t> batch_offsets,
    const int32_t T,
    const int subGroupSize) {
  const auto B = batch_offsets.size(0) - 1;
  const auto num_ads_in_batch = static_cast<uint32_t>(batch_offsets[B]);
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto warp_id =
      (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
      static_cast<uint32_t>(subGroupSize);
  const auto table_id = warp_id / num_ads_in_batch;
  const auto warp_id_in_table = warp_id % num_ads_in_batch;
  const auto num_warp_in_batch = num_ads_in_batch / B;
  const auto batch_id = warp_id_in_table / num_warp_in_batch;
  if (table_id >= T || batch_id >= B) {
    return;
  }

  const auto num_ads_b = batch_offsets[batch_id + 1] - batch_offsets[batch_id];
  const auto output_segment_offset_start =
      table_id * num_ads_in_batch + batch_offsets[batch_id];
  const auto output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];
  const auto input_segment_offset_start = T * batch_id + table_id;
  const auto input_segment_offset_end = input_segment_offset_start + 1;
  const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
  const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
  const auto num_elements = input_segment_end - input_segment_start;

  const auto warp_id_in_batch = warp_id_in_table % num_warp_in_batch;
  const auto lane_id_in_warp = item.get_local_id(0) % subGroupSize;
  for (auto i = warp_id_in_batch; i < num_ads_b; i += num_warp_in_batch) {
    for (auto j = lane_id_in_warp; j < num_elements; j += subGroupSize) {
      reordered_cat_ad_indices[output_segment_start + i * num_elements + j] =
          cat_ad_indices[input_segment_start + j];
    }
  }
}

template <typename Dtype, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void reorder_batched_ad_indices_kernel_(
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> cat_ad_offsets,
    const GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        cat_ad_indices,
    const GenericPackedTensorAccessor<
        index_t,
        1,
        at::RestrictPtrTraits,
        int32_t> reordered_cat_ad_offsets,
    GenericPackedTensorAccessor<Dtype, 1, at::RestrictPtrTraits, int32_t>
        reordered_cat_ad_indices,
    const GenericPackedTensorAccessor<
        int32_t,
        1,
        at::RestrictPtrTraits,
        int32_t> batch_offsets,
    const int32_t T,
    const bool broadcast_indices) {
  const int32_t B = batch_offsets.size(0) - 1;
  const int32_t num_ads_in_batch = batch_offsets[B];
  auto item = syclext::this_work_item::get_nd_item<2>();
  const auto b_t =
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  const int32_t b = b_t % B;
  const int32_t t = b_t / B;
  if (t >= T) {
    return;
  }

  const auto num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
  const auto output_segment_offset_start =
      t * num_ads_in_batch + batch_offsets[b];
  const auto output_segment_start =
      reordered_cat_ad_offsets[output_segment_offset_start];
  const int32_t input_segment_offset_start =
      broadcast_indices ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
  const int32_t input_segment_offset_end = broadcast_indices
      ? input_segment_offset_start + 1
      : input_segment_offset_start + num_ads_b;
  const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
  const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
  const auto num_elements = input_segment_end - input_segment_start;

  if (broadcast_indices) {
    for (auto i = item.get_local_id(1); i < num_ads_b * num_elements;
         i += item.get_local_range(1)) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i % num_elements];
    }
  } else {
    for (auto i = item.get_local_id(1);
         i < input_segment_end - input_segment_start;
         i += item.get_local_range(1)) {
      reordered_cat_ad_indices[output_segment_start + i] =
          cat_ad_indices[input_segment_start + i];
    }
  }
}

void reorder_batched_ad_indices_xpu_kernel(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    at::Tensor& reordered_cat_ad_indices,
    const int64_t num_ads_in_batch,
    const int64_t B,
    const int64_t T,
    const bool broadcast_indices) {
  const int subGroupSize = syclMaxSubGroupSize();
  if (broadcast_indices && T <= 320 && B < 64) {
    TORCH_CHECK(num_ads_in_batch * T == reordered_cat_ad_offsets.numel() - 1);
    if (B == 1) {
      constexpr auto NUM_WARPS = 16;
      const int workGroupSize = NUM_WARPS * subGroupSize;
      const int global_dim =
          xpu_calc_xblock_count(
              reordered_cat_ad_offsets.numel() - 1, NUM_WARPS) *
          workGroupSize;
      FBGEMM_DISPATCH_ALL_TYPES(
          cat_ad_indices.scalar_type(),
          "narrow_broadcast_indices_kernel_1",
          [&] {
            AT_DISPATCH_INDEX_TYPES(
                cat_ad_offsets.scalar_type(),
                "narrow_broadcast_indices_kernel_2",
                [&] {
                  sycl_kernel_submit<
                      narrow_broadcast_indices_kernel_<scalar_t, index_t>>(
                      sycl::range<1>(global_dim),
                      sycl::range<1>(workGroupSize),
                      getCurrentSYCLQueue(),
                      0,
                      cat_ad_offsets.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      cat_ad_indices.packed_accessor32<
                          scalar_t,
                          1,
                          at::RestrictPtrTraits>(),
                      reordered_cat_ad_indices.packed_accessor32<
                          scalar_t,
                          1,
                          at::RestrictPtrTraits>(),
                      num_ads_in_batch,
                      reordered_cat_ad_offsets.numel() - 1,
                      subGroupSize);
                });
          });
      return;
    } else {
      constexpr auto NUM_WARPS = 16;
      const int workGroupSize = NUM_WARPS * subGroupSize;
      const int global_dim =
          xpu_calc_xblock_count(T * num_ads_in_batch, NUM_WARPS) *
          workGroupSize;
      FBGEMM_DISPATCH_ALL_TYPES(
          cat_ad_indices.scalar_type(),
          "narrow_batched_broadcast_indices_kernel_1",
          [&] {
            AT_DISPATCH_INDEX_TYPES(
                cat_ad_offsets.scalar_type(),
                "narrow_batched_broadcast_indices_kernel_2",
                [&] {
                  sycl_kernel_submit<narrow_batched_broadcast_indices_kernel_<
                      scalar_t,
                      index_t>>(
                      sycl::range<1>(global_dim),
                      sycl::range<1>(workGroupSize),
                      getCurrentSYCLQueue(),
                      0,
                      cat_ad_offsets.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      cat_ad_indices.packed_accessor32<
                          scalar_t,
                          1,
                          at::RestrictPtrTraits>(),
                      reordered_cat_ad_offsets.packed_accessor32<
                          index_t,
                          1,
                          at::RestrictPtrTraits>(),
                      reordered_cat_ad_indices.packed_accessor32<
                          scalar_t,
                          1,
                          at::RestrictPtrTraits>(),
                      batch_offsets.packed_accessor32<
                          int32_t,
                          1,
                          at::RestrictPtrTraits>(),
                      T,
                      subGroupSize);
                });
          });
      return;
    }
  }
  FBGEMM_DISPATCH_ALL_TYPES(
      cat_ad_indices.scalar_type(),
      "reorder_batched_ad_indices_xpu_kernel_1",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            cat_ad_offsets.scalar_type(),
            "reorder_batched_ad_indices_xpu_kernel_2",
            [&] {
              constexpr auto NUM_WARPS = 32;
              const int maxWorkGroupSize = syclDeviceMaxWorkGroupSize();
              auto maxWarpSize = maxWorkGroupSize / NUM_WARPS;
              const int global_dim_y =
                  maxWarpSize < subGroupSize ? maxWarpSize : subGroupSize;
              const int global_dim_x =
                  xpu_calc_xblock_count(B * T, NUM_WARPS) * NUM_WARPS;
              sycl_kernel_submit<
                  reorder_batched_ad_indices_kernel_<scalar_t, index_t>>(
                  sycl::range<2>(global_dim_x, global_dim_y),
                  sycl::range<2>(NUM_WARPS, global_dim_y),
                  getCurrentSYCLQueue(),
                  0,
                  cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  reordered_cat_ad_indices
                      .packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>(),
                  batch_offsets
                      .packed_accessor32<int32_t, 1, at::RestrictPtrTraits>(),
                  T,
                  broadcast_indices);
            });
      });
}

} // namespace native::xpu
} // namespace at
