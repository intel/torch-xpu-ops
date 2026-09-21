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

#include <ATen/core/Array.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

template <typename index_t>
struct EmbeddingBagBackwardSumAvgFunctor {
  auto operator()(index_t a, index_t b) const {
    return a == b;
  }
};

template <
    typename scalar_t,
    typename accscalar_t,
    typename index_t,
    int mode,
    int vec_size,
    typename vec_t,
    typename vec_acc_t,
    typename vec_idx_t,
    bool per_sample_weights_defined,
    bool padding_idx_defined>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void embedding_bag_kernel(
    const index_t* const index,
    const index_t* const offset,
    index_t* const offset2bag,
    index_t* const bag_size,
    index_t* const max_index,
    const scalar_t* const per_sample_weights,
    int64_t index_size,
    int64_t bag_num,
    int64_t vectorized_feature_dim_len,
    index_t padding_idx,
    bool ignore_offsets,
    vec_t* o_vec,
    const vec_t* w_vec,
    vec_idx_t* max_idx_vec,
    index_t fixing_bag_size,
    index_t num_row) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto thread_id = item.get_global_linear_id();
  if (thread_id < bag_num * vectorized_feature_dim_len) {
    auto current_feature = thread_id % vectorized_feature_dim_len;
    auto current_bag = thread_id / vectorized_feature_dim_len;
    index_t start, end;
    bool last_bag = current_bag == bag_num - 1;
    if (!ignore_offsets) {
      start = offset[current_bag];
      end = last_bag ? index_size : offset[current_bag + 1];
    } else {
      start = current_bag * fixing_bag_size;
      end = start + fixing_bag_size;
    }

    vec_acc_t value, value_max;
    vec_idx_t index_max;
    index_t padding_cnt = 0;

#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      value[i] = 0;
    }
    if constexpr (mode == MODE_MAX) {
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        value_max[i] = at::numeric_limits<accscalar_t>::lower_bound();
        index_max[i] = -1;
      }
    }
    index_t index_offset, weight_index;
    vec_t wei_load;
    auto handle_non_padding = [&]() {
      wei_load =
          w_vec[weight_index * vectorized_feature_dim_len + current_feature];

      if constexpr (mode == MODE_SUM) {
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
          if constexpr (per_sample_weights_defined) {
            wei_load[i] *= per_sample_weights[index_offset];
          }
          value[i] += wei_load[i];
        }
      } else if constexpr (mode == MODE_MEAN) {
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
          value[i] += wei_load[i];
        }
      } else if constexpr (mode == MODE_MAX) {
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
          if (wei_load[i] > value_max[i]) {
            value_max[i] = wei_load[i];
            if (max_index) {
              index_max[i] = weight_index;
            }
          }
        }
      }
    };

    for (index_t offset_in_bag = start; offset_in_bag < end; offset_in_bag++) {
      index_offset = offset_in_bag;
      weight_index = index[index_offset];
      SYCL_KERNEL_ASSERT(weight_index < num_row);

      if (current_feature == 0)
        offset2bag[index_offset] = current_bag;

      if constexpr (padding_idx_defined) {
        if (padding_idx != weight_index) {
          handle_non_padding();
        } else {
          padding_cnt++;
        }
      } else {
        handle_non_padding();
      }
    }

    int64_t bsize = end - start - padding_cnt;
    if (current_feature == 0) {
      bag_size[current_bag] = bsize;
    }

    index_t o_off = current_bag * vectorized_feature_dim_len + current_feature;
    if constexpr (mode == MODE_SUM) {
      vec_t o;
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        o[i] = value[i];
      }
      o_vec[o_off] = o;
    } else if constexpr (mode == MODE_MEAN) {
      vec_t o;
      bsize = bsize == 0 ? 1 : bsize;
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        o[i] = value[i] / bsize;
      }
      o_vec[o_off] = o;
    } else if constexpr (mode == MODE_MAX) {
      vec_t padding;
#pragma unroll
      for (int i = 0; i < vec_size; i++) {
        padding[i] = 0;
      }
      o_vec[o_off] =
          value_max[0] == at::numeric_limits<accscalar_t>::lower_bound()
          ? padding
          : value_max;
      if (max_index) {
        max_idx_vec[o_off] = index_max;
      }
    }
  }
}

template <typename scalar_t, typename index_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void embedding_bag_per_sample_weights_backward_kernel(
    const scalar_t* grad,
    int64_t grad_stride0,
    int64_t grad_stride1,
    const scalar_t* weight,
    int64_t weight_stride0,
    int64_t weight_stride1,
    const index_t* indices,
    const index_t* offset2bag,
    int64_t num_samples,
    int64_t embedding_features,
    scalar_t* output,
    index_t padding_idx,
    int64_t num_group,
    int64_t max_group_size) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int idx = item.get_global_linear_id();
  auto sg = item.get_sub_group();
  int sgSize = static_cast<int>(
      sg.get_local_range()[0]); // number of work-items in this sub-group
  int sgId = idx / sgSize; // subgroup index
  int sglid = static_cast<int>(
      sg.get_local_id()[0]); // index of the work-item in this sub-group

  int num_sg = num_group * max_group_size / sgSize; // number of sub-groups
  for (int sample_idx = sgId; sample_idx < num_samples; sample_idx += num_sg) {
    accscalar_t result = 0.;
    const int bag_idx = (int)offset2bag[sample_idx];
    const int embedding_idx = (int)indices[sample_idx];
    if (embedding_idx != padding_idx) {
      for (int feature_idx = sglid; feature_idx < embedding_features;
           feature_idx += sgSize) {
        result += grad[grad_stride0 * bag_idx + grad_stride1 * feature_idx] *
            weight[weight_stride0 * embedding_idx +
                   weight_stride1 * feature_idx];
      }
    }
    // subgroup reduce sum
    for (int offset = sgSize / 2; offset > 0; offset /= 2) {
      result += sycl::shift_group_left(sg, result, offset);
    };
    if (sglid == 0) {
      output[sample_idx] = result;
    }
  }
}

} // namespace at::native::xpu
