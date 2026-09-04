/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/ceil_div.h>
#include <ATen/native/xpu/sycl/NMSKernel.h>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace at {
namespace native {
namespace xpu {

constexpr int nms_items_per_group = sizeof(unsigned long long) * 8;

template <typename scalar_t>
inline bool dev_iou(
    scalar_t const* const a,
    scalar_t const* const b,
    const float threshold) {
  scalar_t left = std::max(a[0], b[0]), right = std::min(a[2], b[2]);
  scalar_t top = std::max(a[1], b[1]), bottom = std::min(a[3], b[3]);
  scalar_t width = std::max(right - left, (scalar_t)0),
           height = std::max(bottom - top, (scalar_t)0);
  using acc_t = acc_type_device<scalar_t, kXPU>;
  acc_t area_inter = (acc_t)width * height;
  acc_t area_a = ((acc_t)a[2] - a[0]) * (a[3] - a[1]);
  acc_t area_b = ((acc_t)b[2] - b[0]) * (b[3] - b[1]);
  return (area_inter / (area_a + area_b - area_inter)) > threshold;
}

template <typename scalar_t, typename acc_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void nms_sub_kernel(
    int dets_num,
    float iou_threshold,
    const scalar_t* dets_sorted_ptr,
    unsigned long long* mask_ptr) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  int row_start = item.get_group(0);
  int col_start = item.get_group(1);

  if (row_start > col_start)
    return;

  const int row_size =
      std::min(dets_num - row_start * nms_items_per_group, nms_items_per_group);
  const int col_size =
      std::min(dets_num - col_start * nms_items_per_group, nms_items_per_group);

  char* slm = (char*)syclexp::get_work_group_scratch_memory();
  auto block_boxes = reinterpret_cast<scalar_t*>(slm);
  if (item.get_local_id(1) < col_size) {
    block_boxes[item.get_local_id(1) * 4 + 0] = dets_sorted_ptr
        [(nms_items_per_group * col_start + item.get_local_id(1)) * 4 + 0];
    block_boxes[item.get_local_id(1) * 4 + 1] = dets_sorted_ptr
        [(nms_items_per_group * col_start + item.get_local_id(1)) * 4 + 1];
    block_boxes[item.get_local_id(1) * 4 + 2] = dets_sorted_ptr
        [(nms_items_per_group * col_start + item.get_local_id(1)) * 4 + 2];
    block_boxes[item.get_local_id(1) * 4 + 3] = dets_sorted_ptr
        [(nms_items_per_group * col_start + item.get_local_id(1)) * 4 + 3];
  }
  sycl::group_barrier(item.get_group());

  if (item.get_local_id(1) < row_size) {
    const int cur_box_idx =
        nms_items_per_group * row_start + item.get_local_id(1);
    const scalar_t* cur_box = dets_sorted_ptr + cur_box_idx * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = item.get_local_id(1) + 1;
    }
    for (i = start; i < col_size; i++) {
      if (dev_iou<scalar_t>(cur_box, block_boxes + i * 4, iou_threshold)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks =
        (dets_num + nms_items_per_group - 1) / nms_items_per_group;
    mask_ptr[cur_box_idx * col_blocks + col_start] = t;
  }
}

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void gather_keep_from_mask_kernel(
    bool* keep,
    const unsigned long long* dev_mask,
    const int n_boxes,
    const int col_blocks) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const int thread_id = item.get_local_id(0);

  char* slm = (char*)syclexp::get_work_group_scratch_memory();
  auto removed = reinterpret_cast<unsigned long long*>(slm);
  // Initialize removed
  for (int i = thread_id; i < col_blocks; i += nms_items_per_group) {
    removed[i] = 0;
  }
  sycl::group_barrier(item.get_group());

  for (int nblock = 0; nblock < col_blocks; nblock++) {
    auto removed_val = removed[nblock];
    sycl::group_barrier(item.get_group());
    const int i_offset = nblock * nms_items_per_group;

    for (int inblock = 0; inblock < nms_items_per_group; inblock++) {
      const int i = i_offset + inblock;
      if (i >= n_boxes)
        break;

      // Select a candidate, check if it should be kept
      if (!(removed_val & (1ULL << inblock))) {
        if (thread_id == 0) {
          keep[i] = true;
        }
        auto p = dev_mask + i * col_blocks;

        // Remove all bboxes which overlap the candidate
        for (int j = thread_id; j < col_blocks; j += nms_items_per_group) {
          if (j >= nblock)
            removed[j] |= p[j];
        }
        sycl::group_barrier(item.get_group());
        removed_val = removed[nblock];
      }
    }
  }
}

Tensor nms_kernel(const Tensor& dets_sorted, float iou_threshold) {
  int dets_num = dets_sorted.size(0);
  int col_blocks = ceil_div(dets_num, nms_items_per_group);
  auto mask = at::empty(
      {dets_num * col_blocks}, dets_sorted.options().dtype(at::kLong));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      dets_sorted.scalar_type(),
      "nms_kernel",
      [&] {
        sycl::range<2> global_range{
            (size_t)col_blocks, (size_t)col_blocks * nms_items_per_group};
        sycl::range<2> local_range{1, (size_t)nms_items_per_group};
        using acc_t = acc_type_device<scalar_t, kXPU>;
        auto dets_sorted_ptr = dets_sorted.const_data_ptr<scalar_t>();
        auto mask_ptr = (unsigned long long*)mask.data_ptr<int64_t>();
        constexpr auto kernelFunc = nms_sub_kernel<scalar_t, acc_t>;
        sycl_kernel_submit<kernelFunc>(
            global_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            nms_items_per_group * 4 * sizeof(acc_t),
            dets_num,
            iou_threshold,
            dets_sorted_ptr,
            mask_ptr);
      });

  at::Tensor keep = at::zeros(
      {dets_num}, dets_sorted.options().dtype(at::kBool).device(at::kXPU));
  constexpr auto kernelFunc = gather_keep_from_mask_kernel;
  sycl_kernel_submit<kernelFunc>(
      std::min(col_blocks, nms_items_per_group),
      std::min(col_blocks, nms_items_per_group),
      at::xpu::getCurrentSYCLQueue(),
      col_blocks * sizeof(unsigned long long),
      keep.data_ptr<bool>(),
      (unsigned long long*)mask.data_ptr<int64_t>(),
      dets_num,
      col_blocks);
  return keep;
}

} // namespace xpu
} // namespace native
} // namespace at
