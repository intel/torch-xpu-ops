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

#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

using namespace at::xpu;
constexpr int64_t NROWS_PER_THREAD = 64;

template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void krn_partials_per_segment_kernel(
    index_t* ret_data,
    const index_t* offsets_data,
    const int64_t* num_of_segments_ptr,
    int64_t numel) {
  auto ret_ptr = ret_data;
  auto offsets_ptr = offsets_data;

  auto item = syclext::this_work_item::get_nd_item<1>();
  auto id = item.get_global_linear_id();
  const auto stride = item.get_global_range(0);
  for (; id < *num_of_segments_ptr; id += stride) {
    const index_t idx_start = offsets_ptr[id];
    const index_t idx_end = (id == *num_of_segments_ptr - 1)
        ? static_cast<index_t>(numel)
        : offsets_ptr[id + 1];
    const index_t size = idx_end - idx_start;
    ret_ptr[id] = at::ceil_div(size, static_cast<index_t>(NROWS_PER_THREAD));
  }
}

template <typename index_t>
void krn_partials_per_segment(
    index_t* ret,
    const index_t* segment_offsets,
    const int64_t* num_of_segments_ptr,
    int64_t max_segments,
    int64_t numel) {
  constexpr auto caller = krn_partials_per_segment_kernel<index_t>;
  const int64_t group_size = std::min<int64_t>(
      max_segments, at::xpu::getKernelMaxWorkGroupSize<caller>());
  const int64_t global_range =
      xpuKernelLoopGroupRange(static_cast<int64_t>(max_segments), group_size) *
      group_size;
  sycl_kernel_submit<caller>(
      sycl::range<1>((size_t)global_range),
      sycl::range<1>((size_t)group_size),
      getCurrentSYCLQueue(),
      0,
      ret,
      segment_offsets,
      num_of_segments_ptr,
      numel);
}

template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void krn_partial_segment_offset_kernel(
    index_t* ret_data,
    const index_t* partials_per_segment_data,
    const index_t* partials_per_segment_offset_data,
    const index_t* segment_offsets_data,
    const int64_t* num_of_segments_ptr) {
  auto ret_ptr = ret_data;
  auto partials_per_segment_ptr = partials_per_segment_data;
  auto partials_per_segment_offset_ptr = partials_per_segment_offset_data;
  auto segment_offsets_ptr = segment_offsets_data;

  auto item = syclext::this_work_item::get_nd_item<1>();
  auto id = item.get_global_linear_id();
  auto stride = item.get_global_range(0);
  for (; id < *num_of_segments_ptr; id += stride) {
    index_t idx = partials_per_segment_offset_ptr[id];
    const index_t num_partials = partials_per_segment_ptr[id];
    const index_t segment_offset = segment_offsets_ptr[id];
    for (index_t i = 0; i < num_partials; ++i) {
      ret_ptr[idx++] = segment_offset + i * NROWS_PER_THREAD;
    }
  }
}

template <typename index_t>
void krn_partial_segment_offset(
    index_t* ret,
    const index_t* partials_per_segment,
    const index_t* partials_per_segment_offset,
    const index_t* segment_offsets,
    const int64_t* num_of_segments_ptr,
    int64_t max_segments) {
  constexpr auto caller = krn_partial_segment_offset_kernel<index_t>;
  const int64_t group_size = std::min<int64_t>(
      static_cast<int64_t>(max_segments),
      at::xpu::getKernelMaxWorkGroupSize<caller>());
  const int64_t global_range =
      xpuKernelLoopGroupRange(static_cast<int64_t>(max_segments), group_size) *
      group_size;
  sycl_kernel_submit<caller>(
      sycl::range<1>((size_t)global_range),
      sycl::range<1>((size_t)group_size),
      getCurrentSYCLQueue(),
      0,
      ret,
      partials_per_segment,
      partials_per_segment_offset,
      segment_offsets,
      num_of_segments_ptr);
}

template <typename scalar_t, typename index_t, typename acc_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void compute_grad_weight_bags_kernel(
    int64_t numel,
    int64_t stride,
    int mode_mean,
    int64_t stride_warped,
    bool per_sample_weight_defined,
    bool count_defined,
    int64_t per_sample_weights_stride,
    acc_t* grad_weight_per_segment_data,
    const int64_t* num_of_segments_ptr,
    const index_t* indices_data,
    const scalar_t* gradoutput_data,
    const index_t* offset2bag_data,
    const index_t* count_data,
    const index_t* bag_size_data,
    const scalar_t* per_sample_weights_data,
    const index_t* segment_offsets_data) {
  auto grad_weight_per_segment_ptr = grad_weight_per_segment_data;
  auto indices_ptr = indices_data;
  auto gradoutput_ptr = gradoutput_data;
  auto offset2bag_ptr = offset2bag_data;
  auto count_ptr = count_defined ? count_data : NULL;
  auto bag_size_ptr = bag_size_data;
  auto per_sample_weights_ptr =
      per_sample_weight_defined ? per_sample_weights_data : NULL;
  auto segment_offsets_ptr = segment_offsets_data;

  auto item = syclext::this_work_item::get_nd_item<1>();

  const int gid = item.get_global_linear_id();
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  const auto num_of_segments = *num_of_segments_ptr;
  if (id >= num_of_segments) {
    return;
  }

  const int idx_begin = segment_offsets_ptr[id];
  const int idx_end =
      (id == num_of_segments - 1) ? numel : segment_offsets_ptr[id + 1];

  acc_type_device<scalar_t, kXPU> weight = 0.f;
  for (int idx = idx_begin; idx < idx_end; ++idx) {
    const int orig_row = indices_ptr[idx];
    const int seq_number = offset2bag_ptr[orig_row];
    const int grad_output_row = seq_number * stride;

    acc_type_device<scalar_t, kXPU> scale =
        count_ptr ? 1.f / count_ptr[idx] : 1.f;
    if (per_sample_weight_defined) {
      scale *= per_sample_weights_ptr[orig_row * per_sample_weights_stride];
    }

    acc_type_device<scalar_t, kXPU> gradient =
        gradoutput_ptr[grad_output_row + startFeature];
    if (mode_mean) {
      gradient /= bag_size_ptr[seq_number];
    }
    weight += gradient * scale;
  }
  grad_weight_per_segment_ptr[id * stride + startFeature] = weight;
}

template <typename scalar_t, typename index_t>
void compute_grad_weight_bags(
    const Tensor& indices,
    const Tensor& gradOutput,
    const Tensor& offset2bag,
    const Tensor& count,
    int64_t numel,
    int64_t stride,
    int mode_mean,
    const Tensor& bag_size,
    const Tensor& per_sample_weights,
    const Tensor& segment_offsets,
    const int64_t* num_of_segments_ptr,
    int64_t max_segments,
    const Tensor& grad_weight_per_segment) {
  bool per_sample_weight_defined = per_sample_weights.defined();
  bool count_defined = count.defined();
  int64_t per_sample_weights_stride =
      per_sample_weights.defined() ? per_sample_weights.stride(0) : 0;

  auto grad_weight_per_segment_data =
      grad_weight_per_segment
          .template data_ptr<acc_type_device<scalar_t, kXPU>>();
  auto indices_data = indices.template const_data_ptr<index_t>();
  auto gradOutput_data = gradOutput.const_data_ptr<scalar_t>();
  auto offset2bag_data = offset2bag.const_data_ptr<index_t>();
  auto count_data = count_defined
      ? count.const_data_ptr<index_t>()
      : offset2bag_data; // use the offset2bag_data handler as the dummy
                         // buffer.
  auto bag_size_data = bag_size.const_data_ptr<index_t>();
  auto per_sample_weights_data = per_sample_weight_defined
      ? per_sample_weights.const_data_ptr<scalar_t>()
      : gradOutput_data; // ise the gradOutput_data handler as the dummy
                         // buffer.
  auto segment_offsets_data = segment_offsets.const_data_ptr<index_t>();

  int64_t max_sub_group_size = at::xpu::getDeviceMaxSubGroupSize();
  int64_t stride_warped =
      at::ceil_div(stride, max_sub_group_size) * max_sub_group_size;
  using acc_t = acc_type_device<scalar_t, kXPU>;

  constexpr auto kfn =
      compute_grad_weight_bags_kernel<scalar_t, index_t, acc_t>;

  int64_t work_group_size = at::xpu::getKernelMaxWorkGroupSize<kfn>();
  int64_t group_size = std::min(stride_warped, work_group_size);
  auto num_groups = at::ceil_div(max_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;
  auto global_range = sycl::range<1>((size_t)total_items);
  auto local_range = sycl::range<1>((size_t)group_size);

  sycl_kernel_submit<kfn>(
      global_range,
      local_range,
      getCurrentSYCLQueue(),
      0,
      numel,
      stride,
      mode_mean,
      stride_warped,
      per_sample_weight_defined,
      count_defined,
      per_sample_weights_stride,
      grad_weight_per_segment_data,
      num_of_segments_ptr,
      indices_data,
      gradOutput_data,
      offset2bag_data,
      count_data,
      bag_size_data,
      per_sample_weights_data,
      segment_offsets_data);
}

template <typename scalar_t, typename index_t, typename acc_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void compute_grad_weight_kernel(
    ptrdiff_t numel,
    int64_t stride,
    int64_t stride_warped,
    bool count_defined,
    acc_t* grad_weight_per_segment_data,
    const int64_t* num_of_segments_ptr,
    const index_t* indices_data,
    const scalar_t* grad_output_data,
    const index_t* count_data,
    const index_t* segment_offsets_data) {
  auto grad_weight_per_segment_ptr = grad_weight_per_segment_data;
  auto indices_ptr = indices_data;
  auto grad_output_ptr = grad_output_data;
  auto count_ptr = count_defined ? count_data : NULL;
  auto segment_offsets_ptr = segment_offsets_data;

  auto item = syclext::this_work_item::get_nd_item<1>();
  const int gid = item.get_global_linear_id();
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  const auto num_of_segments = *num_of_segments_ptr;
  if (id >= num_of_segments) {
    return;
  }
  const int idx_begin = segment_offsets_ptr[id];
  const int idx_end =
      (id == num_of_segments - 1) ? numel : segment_offsets_ptr[id + 1];

  acc_type_device<scalar_t, kXPU> weight = 0.f;
  for (int idx = idx_begin; idx < idx_end; idx++) {
    const index_t target_row = indices_ptr[idx];

    const acc_type_device<scalar_t, kXPU> scale =
        count_ptr ? 1.f / count_ptr[idx] : 1.f;
    weight += grad_output_ptr[target_row * stride + startFeature] * scale;
  }
  grad_weight_per_segment_ptr[id * stride + startFeature] = weight;
}

template <typename scalar_t, typename index_t>
void compute_grad_weight(
    const Tensor& indices,
    const Tensor& grad_output,
    const Tensor& count,
    ptrdiff_t numel,
    int64_t stride,
    const Tensor& segment_offsets,
    const int64_t* num_of_segments_ptr,
    int64_t max_segments,
    const Tensor& grad_weight_per_segment) {
  bool count_defined = count.defined();

  auto grad_weight_per_segment_data =
      grad_weight_per_segment.data_ptr<acc_type_device<scalar_t, kXPU>>();
  auto indices_data = indices.const_data_ptr<index_t>();
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto count_data = count_defined
      ? count.const_data_ptr<index_t>()
      : indices_data; // use the indices_data handler as the dummy buffer.
  auto segment_offsets_data = segment_offsets.const_data_ptr<index_t>();

  int64_t max_sub_group_size = at::xpu::getDeviceMaxSubGroupSize();
  int64_t stride_warped =
      at::ceil_div(stride, max_sub_group_size) * max_sub_group_size;
  using acc_t = acc_type_device<scalar_t, kXPU>;

  constexpr auto kfn = compute_grad_weight_kernel<scalar_t, index_t, acc_t>;

  int64_t work_group_size = at::xpu::getKernelMaxWorkGroupSize<kfn>();
  int64_t group_size = std::min(stride_warped, work_group_size);
  auto num_groups = at::ceil_div(max_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;
  auto global_range = sycl::range<1>((size_t)total_items);
  auto local_range = sycl::range<1>((size_t)group_size);

  sycl_kernel_submit<kfn>(
      global_range,
      local_range,
      getCurrentSYCLQueue(),
      0,
      numel,
      stride,
      stride_warped,
      count_defined,
      grad_weight_per_segment_data,
      num_of_segments_ptr,
      indices_data,
      grad_output_data,
      count_data,
      segment_offsets_data);
}

template <typename scalar_t, typename index_t, typename acc_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void sum_and_scatter_kernel(
    int64_t stride,
    int64_t num_of_partial_segments,
    const int64_t padding_idx,
    int64_t stride_warped,
    scalar_t* grad_weight_data,
    const int64_t* num_of_segments_ptr,
    const index_t* input_data,
    const index_t* segment_offsets_data,
    const acc_t* grad_weight_per_segment_data,
    const index_t* segment_sizes_offsets_data) {
  auto grad_weight_ptr = grad_weight_data;
  auto input_ptr = input_data;
  auto segment_offsets_ptr = segment_offsets_data;
  auto grad_weight_per_segment_ptr = grad_weight_per_segment_data;
  auto segment_sizes_offsets_ptr = segment_sizes_offsets_data;

  auto item = syclext::this_work_item::get_nd_item<1>();
  const int gid = item.get_global_linear_id();
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  const auto num_of_segments = *num_of_segments_ptr;
  if (id >= num_of_segments) {
    return;
  }

  const int idx_begin = segment_sizes_offsets_ptr[id];
  const int idx_end = (id == num_of_segments - 1)
      ? num_of_partial_segments
      : segment_sizes_offsets_ptr[id + 1];
  acc_type_device<scalar_t, kXPU> weight = 0.f;
  for (int idx = idx_begin; idx < idx_end; idx++) {
    weight += grad_weight_per_segment_ptr[idx * stride + startFeature];
  }

  int64_t target_row = input_ptr[segment_offsets_ptr[id]];
  if (target_row != padding_idx) {
    grad_weight_ptr[target_row * stride + startFeature] = weight;
  }
}

template <typename scalar_t, typename index_t>
void sum_and_scatter(
    const Tensor& input,
    const Tensor& grad_weight,
    int64_t stride,
    const Tensor& segment_offsets,
    const int64_t* num_of_segments_ptr,
    int64_t max_segments,
    const Tensor& grad_weight_per_segment,
    const Tensor& segment_sizes_offsets,
    int64_t num_of_partial_segments,
    const int64_t padding_idx) {
  auto grad_weight_data = grad_weight.data_ptr<scalar_t>();
  auto input_data = input.const_data_ptr<index_t>();
  auto segment_offsets_data = segment_offsets.const_data_ptr<index_t>();
  auto grad_weight_per_segment_data =
      grad_weight_per_segment.const_data_ptr<acc_type_device<scalar_t, kXPU>>();
  auto segment_sizes_offsets_data =
      segment_sizes_offsets.const_data_ptr<index_t>();
  using acc_t = acc_type_device<scalar_t, kXPU>;

  constexpr auto kfn = sum_and_scatter_kernel<scalar_t, index_t, acc_t>;

  int64_t work_group_size = at::xpu::getKernelMaxWorkGroupSize<kfn>();
  int64_t stride_warped =
      at::ceil_div(stride, work_group_size) * work_group_size;

  int64_t group_size = std::min(stride_warped, work_group_size);
  auto num_groups = at::ceil_div(max_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;
  auto global_range = sycl::range<1>((size_t)total_items);
  auto local_range = sycl::range<1>((size_t)group_size);

  sycl_kernel_submit<kfn>(
      global_range,
      local_range,
      getCurrentSYCLQueue(),
      0,
      stride,
      num_of_partial_segments,
      padding_idx,
      stride_warped,
      grad_weight_data,
      num_of_segments_ptr,
      input_data,
      segment_offsets_data,
      grad_weight_per_segment_data,
      segment_sizes_offsets_data);
}

template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void embedding_segment_flags_kernel(
    const index_t* sorted_indices,
    index_t* flags,
    int64_t numel) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto id = item.get_global_linear_id();
  if (id < numel) {
    flags[id] = id == 0 || sorted_indices[id] != sorted_indices[id - 1];
  }
}

template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void embedding_segment_offsets_kernel(
    const index_t* segment_ids,
    index_t* segment_offsets,
    int64_t* num_of_segments,
    int64_t numel) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto id = item.get_global_linear_id();
  if (id >= numel) {
    return;
  }
  if (id == 0 || segment_ids[id] != segment_ids[id - 1]) {
    segment_offsets[segment_ids[id] - 1] = id;
  }
  if (id == numel - 1) {
    *num_of_segments = segment_ids[id];
  }
}

template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
inline void compute_num_of_partial_segments_kernel(
    const index_t* partials_per_segment,
    const index_t* partials_per_segment_offset,
    const int64_t* num_of_segments,
    int64_t* num_of_partial_segments) {
  const auto last = *num_of_segments - 1;
  *num_of_partial_segments = static_cast<int64_t>(partials_per_segment[last]) +
      partials_per_segment_offset[last];
}

template <typename scalar_t, typename index_t>
Tensor embedding_backward_deterministic_kernel(
    const Tensor& grad,
    const Tensor& orig_indices,
    const Tensor& sorted_indices,
    const Tensor& count,
    int64_t num_weights,
    int64_t padding_idx = -1,
    bool mode_mean = false,
    const Tensor& offset2bag = Tensor(),
    const Tensor& bag_size = Tensor(),
    const Tensor& per_sample_weights = Tensor()) {
  const int64_t numel = sorted_indices.numel();
  auto grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());
  const int64_t stride = grad_weight.stride(0);
  if (numel == 0 || grad.size(-1) == 0) {
    return grad_weight;
  }

  auto segment_offsets = at::empty({numel}, orig_indices.options());
  auto num_of_segments = at::empty({}, orig_indices.options().dtype(kLong));
  auto num_of_segments_ptr = num_of_segments.data_ptr<int64_t>();
  auto& queue = getCurrentSYCLQueue();
  {
    auto flags = at::empty({numel}, orig_indices.options());
    auto segment_ids = at::empty_like(flags);
    auto flags_ptr = flags.data_ptr<index_t>();
    auto segment_ids_ptr = segment_ids.data_ptr<index_t>();
    constexpr auto flags_kfn = embedding_segment_flags_kernel<index_t>;
    const int64_t flags_group_size = std::min<int64_t>(
        numel, at::xpu::getKernelMaxWorkGroupSize<flags_kfn>());
    const int64_t flags_global_range =
        at::ceil_div(numel, flags_group_size) * flags_group_size;
    sycl_kernel_submit<flags_kfn>(
        sycl::range<1>((size_t)flags_global_range),
        sycl::range<1>((size_t)flags_group_size),
        queue,
        0,
        sorted_indices.const_data_ptr<index_t>(),
        flags_ptr,
        numel);
    pstl::inclusive_scan(
        flags_ptr, flags_ptr + numel, segment_ids_ptr, (index_t)0);
    constexpr auto offsets_kfn = embedding_segment_offsets_kernel<index_t>;
    const int64_t offsets_group_size = std::min<int64_t>(
        numel, at::xpu::getKernelMaxWorkGroupSize<offsets_kfn>());
    const int64_t offsets_global_range =
        at::ceil_div(numel, offsets_group_size) * offsets_group_size;
    sycl_kernel_submit<offsets_kfn>(
        sycl::range<1>((size_t)offsets_global_range),
        sycl::range<1>((size_t)offsets_group_size),
        queue,
        0,
        segment_ids_ptr,
        segment_offsets.data_ptr<index_t>(),
        num_of_segments_ptr,
        numel);
  }

  const int64_t max_segments = std::min(numel, num_weights);
  auto partials_per_segment = at::empty({max_segments}, orig_indices.options());

  krn_partials_per_segment<index_t>(
      partials_per_segment.template data_ptr<index_t>(),
      segment_offsets.const_data_ptr<index_t>(),
      num_of_segments_ptr,
      max_segments,
      numel);

  // In order to compute `partial_segment_offset`, which is the start index
  // of each partial-segment in `sorted_indices`, we need to compute the
  // start position of each _segment_ in `partial_segment_offset`.
  // Unit: index in `partial_segment_offset`
  auto partials_per_segment_offset =
      at::empty({max_segments}, orig_indices.options());
  pstl::exclusive_scan(
      partials_per_segment.template data_ptr<index_t>(),
      partials_per_segment.template data_ptr<index_t>() + max_segments,
      partials_per_segment_offset.template data_ptr<index_t>(),
      (index_t)0);

  // The total number of partial-segments is the sum of
  // `partials_per_segment_offset`
  auto num_of_partial_segments =
      at::empty({}, orig_indices.options().dtype(kLong));
  auto num_of_partial_segments_ptr =
      num_of_partial_segments.data_ptr<int64_t>();
  constexpr auto num_partials_kfn =
      compute_num_of_partial_segments_kernel<index_t>;
  sycl_kernel_submit<num_partials_kfn>(
      sycl::range<1>(1),
      sycl::range<1>(1),
      queue,
      0,
      partials_per_segment.const_data_ptr<index_t>(),
      partials_per_segment_offset.const_data_ptr<index_t>(),
      num_of_segments_ptr,
      num_of_partial_segments_ptr);

  const int64_t max_partial_segments = numel / NROWS_PER_THREAD + max_segments;
  auto partial_segment_offset =
      at::empty({max_partial_segments}, orig_indices.options());
  krn_partial_segment_offset<index_t>(
      partial_segment_offset.template data_ptr<index_t>(),
      partials_per_segment.template data_ptr<index_t>(),
      partials_per_segment_offset.template data_ptr<index_t>(),
      segment_offsets.const_data_ptr<index_t>(),
      num_of_segments_ptr,
      max_segments);

  TensorOptions op;
  if (grad.dtype() == at::kBFloat16 || grad.dtype() == at::kHalf) {
    op = grad.options().dtype(at::kFloat);
  } else {
    op = grad.options();
  }
  auto grad_weight_per_segment = at::empty({max_partial_segments, stride}, op);
  // Compute the sum of each partial-segment and handle bags
  if (offset2bag.defined()) {
    compute_grad_weight_bags<scalar_t, index_t>(
        orig_indices,
        grad,
        offset2bag,
        count,
        numel,
        stride,
        mode_mean,
        bag_size,
        per_sample_weights,
        partial_segment_offset,
        num_of_partial_segments_ptr,
        max_partial_segments,
        grad_weight_per_segment);
  } else {
    compute_grad_weight<scalar_t, index_t>(
        orig_indices,
        grad,
        count,
        numel,
        stride,
        partial_segment_offset,
        num_of_partial_segments_ptr,
        max_partial_segments,
        grad_weight_per_segment);
  }

  sum_and_scatter<scalar_t, index_t>(
      sorted_indices,
      grad_weight,
      stride,
      segment_offsets,
      num_of_segments_ptr,
      max_segments,
      grad_weight_per_segment,
      partials_per_segment_offset,
      num_of_partial_segments_ptr,
      padding_idx);

  return grad_weight;
}

} // namespace at::native::xpu
