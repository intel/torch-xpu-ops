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
    typename vec_idx_t>
struct EmbeddingBagKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);
    index_t start = 0, end = 0;
    int64_t off_off = -1;

    do {
      if (desc.glb_problem < cfg_.problem_ &&
          desc.glb_batch < cfg_.problem_batch_) {
        bool walk_on_bag = desc.glb_batch != off_off;
        if (walk_on_bag) {
          off_off = desc.glb_batch;
          bool last_bag = off_off == bag_num_ - 1;
          if (!ignore_offsets_) {
            start = offset_[off_off];
            end = last_bag ? index_size_ : offset_[off_off + 1];
          } else {
            start = off_off * fixing_bag_size_;
            end = start + fixing_bag_size_;
          }
        }

        vec_acc_t value, value_max;
        vec_idx_t index_max;
        index_t padding_cnt = 0;
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
          value[i] = 0;
          value_max[i] = at::numeric_limits<accscalar_t>::lower_bound();
          index_max[i] = -1;
        }

        for (index_t off = start; off < end; off++) {
          index_t index_off = off;
          index_t vec_idx = index_[index_off];
          SYCL_KERNEL_ASSERT(vec_idx < num_row_);

          if (walk_on_bag && desc.glb_problem == 0) {
            offset2bag_[index_off] = off_off;
          }

          if (padding_idx_ != vec_idx) {
            index_t i_off = vec_idx * vec_len_ + desc.glb_problem;
            vec_t other = w_vec_[i_off];

            if constexpr (mode == MODE_SUM) {
#pragma unroll
              for (int i = 0; i < vec_size; i++) {
                if (per_sample_weights_) {
                  other[i] *= per_sample_weights_[index_off];
                }
                value[i] += other[i];
              }
            } else if constexpr (mode == MODE_MEAN) {
#pragma unroll
              for (int i = 0; i < vec_size; i++) {
                value[i] += other[i];
              }
            } else if constexpr (mode == MODE_MAX) {
#pragma unroll
              for (int i = 0; i < vec_size; i++) {
                if (other[i] > value_max[i]) {
                  value_max[i] = other[i];
                  if (max_index_) {
                    index_max[i] = vec_idx;
                  }
                }
              }
            }
          } else {
            padding_cnt++;
          }
        }

        int64_t bsize = end - start - padding_cnt;
        if (desc.glb_problem == 0) {
          bag_size_[off_off] = bsize;
        }

        index_t o_off = off_off * vec_len_ + desc.glb_problem;
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
    } while (cfg_.next(item, desc));
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
      int64_t vec_len,
      index_t padding_idx,
      bool ignore_offsets,
      vec_t* o_vec,
      const vec_t* w_vec,
      vec_idx_t* max_idx_vec,
      BatchKernelConfig cfg,
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
        vec_len_(vec_len),
        padding_idx_(padding_idx),
        ignore_offsets_(ignore_offsets),
        o_vec_(o_vec),
        w_vec_(w_vec),
        max_idx_vec_(max_idx_vec),
        cfg_(cfg),
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
  int64_t vec_len_;
  index_t padding_idx_;
  bool ignore_offsets_;
  vec_t* o_vec_;
  const vec_t* w_vec_;
  vec_idx_t* max_idx_vec_;
  BatchKernelConfig cfg_;
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
