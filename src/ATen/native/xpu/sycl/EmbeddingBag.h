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
struct EmbeddingBagKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto thread_id = item.get_global_linear_id();
    if (thread_id < bag_num_ * vectorized_feature_dim_len_) {
      auto current_feature = thread_id % vectorized_feature_dim_len_;
      auto current_bag = thread_id / vectorized_feature_dim_len_;
      index_t start, end;
      bool last_bag = current_bag == bag_num_ - 1;
      if (!ignore_offsets_) {
        start = offset_[current_bag];
        end = last_bag ? index_size_ : offset_[current_bag + 1];
      } else {
        start = current_bag * fixing_bag_size_;
        end = start + fixing_bag_size_;
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
        wei_load = w_vec_
            [weight_index * vectorized_feature_dim_len_ + current_feature];

        if constexpr (mode == MODE_SUM) {
#pragma unroll
          for (int i = 0; i < vec_size; i++) {
            if constexpr (per_sample_weights_defined) {
              wei_load[i] *= per_sample_weights_[index_offset];
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
              if (max_index_) {
                index_max[i] = weight_index;
              }
            }
          }
        }
      };

      for (index_t offset_in_bag = start; offset_in_bag < end;
           offset_in_bag++) {
        index_offset = offset_in_bag;
        weight_index = index_[index_offset];
        SYCL_KERNEL_ASSERT(weight_index < num_row_);

        if (current_feature == 0)
          offset2bag_[index_offset] = current_bag;

        if constexpr (padding_idx_defined) {
          if (padding_idx_ != weight_index) {
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
        bag_size_[current_bag] = bsize;
      }

      index_t o_off =
          current_bag * vectorized_feature_dim_len_ + current_feature;
      if constexpr (mode == MODE_SUM) {
        vec_t o;
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
          o[i] = value[i];
        }
        o_vec_[o_off] = o;
      } else if constexpr (mode == MODE_MEAN) {
        vec_t o;
        bsize = bsize == 0 ? 1 : bsize;
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
          o[i] = value[i] / bsize;
        }
        o_vec_[o_off] = o;
      } else if constexpr (mode == MODE_MAX) {
        vec_t padding;
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
          padding[i] = 0;
        }
        o_vec_[o_off] =
            value_max[0] == at::numeric_limits<accscalar_t>::lower_bound()
            ? padding
            : value_max;
        if (max_index_) {
          max_idx_vec_[o_off] = index_max;
        }
      }
    }
  }
  EmbeddingBagKernelFunctor(
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
      index_t num_row)
      : index_(index),
        offset_(offset),
        offset2bag_(offset2bag),
        bag_size_(bag_size),
        max_index_(max_index),
        per_sample_weights_(per_sample_weights),
        index_size_(index_size),
        bag_num_(bag_num),
        vectorized_feature_dim_len_(vectorized_feature_dim_len),
        padding_idx_(padding_idx),
        ignore_offsets_(ignore_offsets),
        o_vec_(o_vec),
        w_vec_(w_vec),
        max_idx_vec_(max_idx_vec),
        fixing_bag_size_(fixing_bag_size),
        num_row_(num_row) {}

 private:
  const index_t* const index_;
  const index_t* const offset_;
  index_t* const offset2bag_;
  index_t* const bag_size_;
  index_t* const max_index_;
  const scalar_t* const per_sample_weights_;
  int64_t index_size_;
  int64_t bag_num_;
  int64_t vectorized_feature_dim_len_;
  index_t padding_idx_;
  bool ignore_offsets_;
  vec_t* o_vec_;
  const vec_t* w_vec_;
  vec_idx_t* max_idx_vec_;
  index_t fixing_bag_size_;
  index_t num_row_;
};

template <typename index_t>
struct EmbeddingBagBackwardSumAvgFunctor {
  auto operator()(index_t a, index_t b) const {
    return a == b;
  }
};

template <typename scalar_t, typename index_t, typename accscalar_t>
struct EmbeddingBagPerSampleWeightsBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item_id) const {
    int idx = item_id.get_global_linear_id();
    auto sg = item_id.get_sub_group();
    int sgSize = static_cast<int>(
        sg.get_local_range()[0]); // number of work-items in this sub-group
    int sgId = idx / sgSize; // subgroup index
    int sglid = static_cast<int>(
        sg.get_local_id()[0]); // index of the work-item in this sub-group

    int num_sg = num_group_ * max_group_size_ / sgSize; // number of sub-groups
    for (int sample_idx = sgId; sample_idx < num_samples_;
         sample_idx += num_sg) {
      accscalar_t result = 0.;
      const int bag_idx = (int)offset2bag_[sample_idx];
      const int embedding_idx = (int)indices_[sample_idx];
      if (embedding_idx != padding_idx_) {
        for (int feature_idx = sglid; feature_idx < embedding_features_;
             feature_idx += sgSize) {
          result +=
              grad_[grad_stride0_ * bag_idx + grad_stride1_ * feature_idx] *
              weight_
                  [weight_stride0_ * embedding_idx +
                   weight_stride1_ * feature_idx];
        }
      }
      // subgroup reduce sum
      for (int offset = sgSize / 2; offset > 0; offset /= 2) {
        result += sycl::shift_group_left(sg, result, offset);
      };
      if (sglid == 0) {
        output_[sample_idx] = result;
      }
    }
  }
  EmbeddingBagPerSampleWeightsBackwardKernelFunctor(
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
      int64_t max_group_size)
      : grad_(grad),
        grad_stride0_(grad_stride0),
        grad_stride1_(grad_stride1),
        weight_(weight),
        weight_stride0_(weight_stride0),
        weight_stride1_(weight_stride1),
        indices_(indices),
        offset2bag_(offset2bag),
        num_samples_(num_samples),
        embedding_features_(embedding_features),
        output_(output),
        padding_idx_(padding_idx),
        num_group_(num_group),
        max_group_size_(max_group_size) {}

 private:
  const scalar_t* grad_;
  int64_t grad_stride0_;
  int64_t grad_stride1_;
  const scalar_t* weight_;
  int64_t weight_stride0_;
  int64_t weight_stride1_;
  const index_t* indices_;
  const index_t* offset2bag_;
  int64_t num_samples_;
  int64_t embedding_features_;
  scalar_t* output_;
  index_t padding_idx_;
  int64_t num_group_;
  int64_t max_group_size_;
};

} // namespace at::native::xpu
