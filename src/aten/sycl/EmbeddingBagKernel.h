#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Array.h>

#include <aten/sycl/BatchKernel.h>
#include <aten/sycl/NumericLimits.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

namespace {

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
      index_t* const index,
      index_t* const offset,
      index_t* const offset2bag,
      index_t* const bag_size,
      index_t* const max_index,
      scalar_t* const per_sample_weights,
      int64_t index_size,
      int64_t bag_num,
      int64_t vec_len,
      index_t padding_idx,
      bool ignore_offsets,
      vec_t* o_vec,
      vec_t* w_vec,
      vec_idx_t* max_idx_vec,
      BatchKernelConfig cfg,
      index_t fixing_bag_size)
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
        fixing_bag_size_(fixing_bag_size) {}

 private:
  index_t* const index_;
  index_t* const offset_;
  index_t* const offset2bag_;
  index_t* const bag_size_;
  index_t* const max_index_;
  scalar_t* const per_sample_weights_;
  int64_t index_size_;
  int64_t bag_num_;
  int64_t vec_len_;
  index_t padding_idx_;
  bool ignore_offsets_;
  vec_t* o_vec_;
  vec_t* w_vec_;
  vec_idx_t* max_idx_vec_;
  BatchKernelConfig cfg_;
  index_t fixing_bag_size_;
};

template <
    int vec_size,
    typename vec_t,
    typename scalar_t,
    typename accscalar_t,
    typename index_t>
void vec_chunk_kernel_embeddingbag(
    const int64_t mode,
    index_t* input,
    index_t* offset,
    scalar_t* weight,
    scalar_t* output,
    index_t* offset2bag,
    index_t* bag_size,
    bool per_sample_weights_defined,
    scalar_t* per_sample_weights,
    int64_t per_sample_weights_stride,
    index_t* max_indices,
    int64_t WGNumber,
    int64_t numBags,
    int64_t weight_total_elem,
    int64_t chunk_size,
    int64_t bag_chunk_num,
    int64_t bag_wi_num,
    int64_t bagsPerLoop,
    int64_t input_length,
    int64_t weight_stride0,
    int64_t weight_stride1,
    const bool include_last_offset,
    const index_t padding_idx,
    const bool if_align_vector,
    sycl::nd_item<1> item) {
  auto globalId = item.get_global_linear_id();

  // global chunk id
  auto globalChunkId = globalId / chunk_size;

  // which initial bag this work item is in
  auto bagId = globalChunkId / bag_chunk_num;

  // work item id inside one bag
  auto insideBagId = globalId % bag_wi_num;

  constexpr int align_bytes = alignof(vec_t);

  // outer bag loop
  for (auto bag = bagId; bag < numBags; bag += bagsPerLoop) {
    auto begin = offset[bag];

    // TODO: Here need a check for begin and end that end must >= begin.
    auto end = (bag < (numBags - 1))
        ? (offset[bag + 1])
        : (include_last_offset ? offset[bag + 1] : input_length);

    // for mean mode's backward
    index_t bag_size_ = 0;

    // In single_bag situation, embeddingbag is like embedding, no
    // per_sample_weight, mode is not max and not padding entry and 2D weight,
    // pure vec copy is used to achieve most memory bandwidth.
    auto single_bag = bool(
        (end == (begin + 1)) && (!per_sample_weights_defined) &&
        (mode != MODE_MAX) && (input[begin] != padding_idx));

    if (single_bag) {
      auto input_single_elem = input[begin];

      // for checking alignment with vector
      auto shift = ((uint64_t)(weight + input_single_elem * weight_stride0)) %
          align_bytes / sizeof(scalar_t);

      // here the shift elements need to be individually dealed with
      for (auto mis_idx = 0; mis_idx < shift; ++mis_idx) {
        if (insideBagId == 0) {
          if (mis_idx < weight_stride0) {
            output[bag * weight_stride0 + mis_idx] = weight
                [input_single_elem * weight_stride0 + mis_idx * weight_stride1];
          }
        }
      }

      if (((shift + input_single_elem * weight_stride0) < weight_total_elem) &&
          (shift < weight_stride0)) {
        vec_t* weight_vec = reinterpret_cast<vec_t*>(
            shift + weight + input_single_elem * weight_stride0);
        // vector load
        auto weightSingleValue = weight_vec[insideBagId];
        vec_t* output_vec =
            reinterpret_cast<vec_t*>(shift + output + bag * weight_stride0);
#pragma unroll
        for (auto id = 0; id < vec_size; id++) {
          if ((shift + insideBagId * vec_size + id) < weight_stride0) {
            output_vec[insideBagId][id] =
                weightSingleValue[id * weight_stride1];
          }
        }
      }

      if (insideBagId == 0) {
        offset2bag[begin] = bag;
        bag_size[bag] = static_cast<index_t>(1);
      }
    } else {
      // not single bag mode
      index_t maxWord[vec_size];
      accscalar_t weightFeatSum[vec_size];
      scalar_t weightFeatMax[vec_size];

#pragma unroll
      for (auto id = 0; id < vec_size; id++) {
        maxWord[id] = -1;
        weightFeatSum[id] = static_cast<accscalar_t>(0.0);
        weightFeatMax[id] = static_cast<scalar_t>(0.0);
      }

      // alignment with vector load
      if (if_align_vector) {
        for (auto emb = begin; emb < end; emb++) {
          auto input_elem = input[emb];

          // if this bag copes with multi embeddings and one of these embeddings
          // is padding_idx, this embedding is ignored for reduction because
          // embedding vector at padding_idx is excluded from the reduction
          bool pad = (input_elem == padding_idx);

          // vector process remaining
          vec_t* weight_vec =
              reinterpret_cast<vec_t*>(weight + input_elem * weight_stride0);
          auto weightValue = weight_vec[insideBagId];

#pragma unroll
          for (auto id = 0; id < vec_size; id++) {
            if ((insideBagId * vec_size + id) < weight_stride0) {
              if (mode == MODE_MAX) {
                // static_cast to scalar_t is used because vec_t contains
                // uint dtype
                auto val = weightValue[id];
                auto max_val = weightFeatMax[id];
                // bag_size_ == 0 means it first come
                if (bag_size_ == 0 || val > max_val) {
                  // padded entry will not be included output
                  weightFeatMax[id] = pad ? weightFeatMax[id] : weightValue[id];
                  maxWord[id] = pad ? maxWord[id] : input_elem;
                }
              } else {
                // 1. for scalar type fma/add, accscalar_t is needed to keep
                // accurate. Vec is stored uint value, whose size is same
                // as sizeof(scalar_t), when computing, uint value should
                // be casted to floating value, after computation,
                // write-back needs casting to uint value.
                // 2. if this entry is padded, 0 value is prepared for
                // reduce(sum/mean)
                auto val = pad ? static_cast<scalar_t>(0.0) : weightValue[id];
                auto acc_val = static_cast<accscalar_t>(val);
                auto acc_sum = weightFeatSum[id];
                if (per_sample_weights_defined) {
                  auto scaleWeightBy = static_cast<accscalar_t>(
                      per_sample_weights[emb * per_sample_weights_stride]);
                  acc_sum += acc_val * scaleWeightBy;
                } else {
                  acc_sum += acc_val;
                }
                weightFeatSum[id] = acc_sum;
              }
            }
          }

          // if this entry is padded, it will not contribute to bag size
          bag_size_ += pad ? 0 : 1;

          // avoid compete write in and padded entry also needs to be recorded
          // to offset2bag
          if (insideBagId == 0) {
            offset2bag[emb] = bag;
          }
        }
      } else {
        // exist misalignment, back to single point processing
        for (auto emb = begin; emb < end; emb++) {
          auto input_elem = input[emb];
          // if this bag copes with multi embeddings and one of these embeddings
          // is padding_idx, this embedding is ignored for reduction because
          // embedding vector at padding_idx is excluded from the reduction
          bool pad = (input_elem == padding_idx);

#pragma unroll
          for (auto id = 0; id < vec_size; id++) {
            if ((insideBagId * vec_size + id) < weight_stride0) {
              auto weight_idx = input_elem * weight_stride0 +
                  insideBagId * vec_size + id * weight_stride1;
              if (mode == MODE_MAX) {
                // static_cast to scalar_t is used because vec_t contains
                // uint dtype
                auto val = weight[weight_idx];
                auto max_val = weightFeatMax[id];
                // bag_size_ == 0 means it first come
                if (bag_size_ == 0 || val > max_val) {
                  // padded entry will not be included output
                  weightFeatMax[id] = pad ? weightFeatMax[id] : val;
                  maxWord[id] = pad ? maxWord[id] : input_elem;
                }
              } else {
                // 1. for scalar type fma/add, accscalar_t is needed to keep
                // accurate. Vec is stored uint value, whose size is same
                // as sizeof(scalar_t), when computing, uint value should
                // be casted to floating value, after computation,
                // write-back needs casting to uint value.
                // 2. if this entry is padded, 0 value is prepared for
                // reduce(sum/mean)
                auto val =
                    pad ? static_cast<scalar_t>(0.0) : weight[weight_idx];
                auto acc_val = static_cast<accscalar_t>(val);
                if (per_sample_weights_defined) {
                  auto scaleWeightBy = static_cast<accscalar_t>(
                      per_sample_weights[emb * per_sample_weights_stride]);
                  weightFeatSum[id] += acc_val * scaleWeightBy;
                } else {
                  weightFeatSum[id] += acc_val;
                }
              }
            }
          }

          // if this entry is padded, it will not contribute to bag size
          bag_size_ += pad ? 0 : 1;

          // avoid compete write in and padded entry also needs to be recorded
          // to offset2bag
          if (insideBagId == 0) {
            offset2bag[emb] = bag;
          }
        }
      }

      // calculate average for mean mode
      if (mode == MODE_MEAN) {
#pragma unroll
        for (auto id = 0; id < vec_size; id++) {
          if ((insideBagId * vec_size + id) < weight_stride0) {
            auto acc_sum = weightFeatSum[id];
            if (bag_size_ != 0) {
              acc_sum /= static_cast<accscalar_t>(bag_size_);
            }
            weightFeatSum[id] = acc_sum;
          }
        }
      }

      // output
#pragma unroll
      for (auto id = 0; id < vec_size; id++) {
        if ((insideBagId * vec_size + id) < weight_stride0) {
          auto output_idx = bag * weight_stride0 + insideBagId * vec_size +
              id * weight_stride1;
          if (mode == MODE_MEAN || mode == MODE_SUM) {
            output[output_idx] = static_cast<scalar_t>(weightFeatSum[id]);
          } else if (mode == MODE_MAX) {
            output[output_idx] = weightFeatMax[id];
            max_indices[output_idx] = maxWord[id];
          }
        }
      }

      if (insideBagId == 0) {
        bag_size[bag] = static_cast<index_t>(bag_size_);
      }
    }
  }
}

template <
    int vec_size,
    typename vec_t,
    typename scalar_t,
    typename accscalar_t,
    typename index_t>
struct EmbeddingBagUpdateOutputKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    vec_chunk_kernel_embeddingbag<
        vec_size,
        vec_t,
        scalar_t,
        accscalar_t,
        index_t>(
        mode_,
        input_,
        offset_,
        weight_,
        output_,
        offset2bag_,
        bag_size_,
        per_sample_weights_defined_,
        per_sample_weights_,
        per_sample_weights_stride_,
        max_indices_,
        WGNumber_,
        numBags_,
        weight_total_elem_,
        chunk_size_,
        bag_chunk_num_,
        bag_wi_num_,
        bagsPerLoop_,
        input_length_,
        weight_stride0_,
        weight_stride1_,
        include_last_offset_,
        padding_idx_,
        if_align_vector_,
        item);
  }
  EmbeddingBagUpdateOutputKernelFunctor(
      const int64_t mode,
      index_t* input,
      index_t* offset,
      scalar_t* weight,
      scalar_t* output,
      index_t* offset2bag,
      index_t* bag_size,
      bool per_sample_weights_defined,
      scalar_t* per_sample_weights,
      int64_t per_sample_weights_stride,
      index_t* max_indices,
      int64_t WGNumber,
      int64_t numBags,
      int64_t weight_total_elem,
      int64_t chunk_size,
      int64_t bag_chunk_num,
      int64_t bag_wi_num,
      int64_t bagsPerLoop,
      int64_t input_length,
      int64_t weight_stride0,
      int64_t weight_stride1,
      bool include_last_offset,
      index_t padding_idx,
      bool if_align_vector)
      : mode_(mode),
        input_(input),
        offset_(offset),
        weight_(weight),
        output_(output),
        offset2bag_(offset2bag),
        bag_size_(bag_size),
        per_sample_weights_defined_(per_sample_weights_defined),
        per_sample_weights_(per_sample_weights),
        per_sample_weights_stride_(per_sample_weights_stride),
        max_indices_(max_indices),
        WGNumber_(WGNumber),
        numBags_(numBags),
        weight_total_elem_(weight_total_elem),
        chunk_size_(chunk_size),
        bag_chunk_num_(bag_chunk_num),
        bag_wi_num_(bag_wi_num),
        bagsPerLoop_(bagsPerLoop),
        input_length_(input_length),
        weight_stride0_(weight_stride0),
        weight_stride1_(weight_stride1),
        include_last_offset_(include_last_offset),
        padding_idx_(padding_idx),
        if_align_vector_(if_align_vector) {}

 private:
  const int64_t mode_;
  index_t* input_;
  index_t* offset_;
  scalar_t* weight_;
  scalar_t* output_;
  index_t* offset2bag_;
  index_t* bag_size_;
  bool per_sample_weights_defined_;
  scalar_t* per_sample_weights_;
  int64_t per_sample_weights_stride_;
  index_t* max_indices_;
  int64_t WGNumber_;
  int64_t numBags_;
  int64_t weight_total_elem_;
  int64_t chunk_size_;
  int64_t bag_chunk_num_;
  int64_t bag_wi_num_;
  int64_t bagsPerLoop_;
  int64_t input_length_;
  int64_t weight_stride0_;
  int64_t weight_stride1_;
  const bool include_last_offset_;
  const index_t padding_idx_;
  const bool if_align_vector_;
};

} // namespace

::std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_kernel(
    const Tensor& weight_t,
    const Tensor& indices_t,
    const Tensor& offsets_t,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights_t,
    bool include_last_offset,
    int64_t padding_idx);

} // namespace at::native::xpu
