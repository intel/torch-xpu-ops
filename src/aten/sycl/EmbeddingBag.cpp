#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>

#include <aten/sycl/Atomics.h>
#include <aten/sycl/EmbeddingBackwardKernel.h>
#include <aten/sycl/EmbeddingBagKernel.h>
#include <aten/sycl/MemoryAccess.h>

using namespace at::native::xpu::detail;

namespace at::native::xpu {

namespace detail {

namespace {

template <typename T, typename V>
inline auto CeilDiv(T a, V b) {
  return (a + b - 1) / b;
}

} // namespace

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? indices
                                          : indices.toType(commonType),
      offsets.scalar_type() == commonType ? offsets
                                          : offsets.toType(commonType)};
}

#define EMBBAG_KERNEL_ACC(                                                   \
    scalar_t,                                                                \
    accscalar_t,                                                             \
    index_t,                                                                 \
    mode,                                                                    \
    vec_size,                                                                \
    output,                                                                  \
    weight,                                                                  \
    input,                                                                   \
    offset,                                                                  \
    offset2bag,                                                              \
    bag_size,                                                                \
    max_indices,                                                             \
    per_sample_weights,                                                      \
    index_len,                                                               \
    bag_num,                                                                 \
    vec_len,                                                                 \
    padding_idx,                                                             \
    ignore_offsets)                                                          \
  embedding_bag_kernel<scalar_t, accscalar_t, index_t, mode, vec_size>(      \
      output.data_ptr<scalar_t>(),                                           \
      weight.data_ptr<scalar_t>(),                                           \
      indices.data_ptr<index_t>(),                                           \
      offsets.data_ptr<index_t>(),                                           \
      offset2bag.data_ptr<index_t>(),                                        \
      bag_size.data_ptr<index_t>(),                                          \
      max_indices.data_ptr<index_t>(),                                       \
      per_sample_weights.defined() ? per_sample_weights.data_ptr<scalar_t>() \
                                   : nullptr,                                \
      index_size,                                                            \
      bag_num,                                                               \
      vec_len,                                                               \
      padding_idx,                                                           \
      ignore_offsets)

#define EMBBAG_KERNEL_NO_ACC(                                                \
    scalar_t,                                                                \
    index_t,                                                                 \
    mode,                                                                    \
    vec_size,                                                                \
    output,                                                                  \
    weight,                                                                  \
    input,                                                                   \
    offset,                                                                  \
    offset2bag,                                                              \
    bag_size,                                                                \
    max_indices,                                                             \
    per_sample_weights,                                                      \
    index_len,                                                               \
    bag_num,                                                                 \
    vec_len,                                                                 \
    padding_idx,                                                             \
    ignore_offsets)                                                          \
  embedding_bag_kernel<scalar_t, scalar_t, index_t, mode, vec_size>(         \
      output.data_ptr<scalar_t>(),                                           \
      weight.data_ptr<scalar_t>(),                                           \
      indices.data_ptr<index_t>(),                                           \
      offsets.data_ptr<index_t>(),                                           \
      offset2bag.data_ptr<index_t>(),                                        \
      bag_size.data_ptr<index_t>(),                                          \
      max_indices.data_ptr<index_t>(),                                       \
      per_sample_weights.defined() ? per_sample_weights.data_ptr<scalar_t>() \
                                   : nullptr,                                \
      index_size,                                                            \
      bag_num,                                                               \
      vec_len,                                                               \
      padding_idx,                                                           \
      ignore_offsets)

template <
    typename scalar_t,
    typename accscalar_t,
    typename index_t,
    int mode,
    int vec_size>
void embedding_bag_kernel(
    scalar_t* const output,
    scalar_t* const weights,
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
    bool ignore_offsets) {
  using vec_t = at::detail::Array<scalar_t, vec_size>;
  using vec_acc_t = at::detail::Array<accscalar_t, vec_size>;
  using vec_idx_t = at::detail::Array<index_t, vec_size>;

  vec_t* o_vec = reinterpret_cast<vec_t*>(output);
  vec_t* w_vec = reinterpret_cast<vec_t*>(weights);
  vec_idx_t* max_idx_vec = reinterpret_cast<vec_idx_t*>(max_index);

  vec_len = vec_len / vec_size;
  BatchKernelConfig cfg = {
      bag_num, vec_len, 1, bag_num, true, BatchKernelConfig::Policy::pAdaptive};
  index_t fixing_bag_size = ignore_offsets ? index_size / bag_num : 0;
  auto caller = EmbeddingBagKernelFunctor<
      scalar_t,
      accscalar_t,
      index_t,
      mode,
      vec_size,
      vec_t,
      vec_acc_t,
      vec_idx_t>(
      index,
      offset,
      offset2bag,
      bag_size,
      max_index,
      per_sample_weights,
      index_size,
      bag_num,
      vec_len,
      padding_idx,
      ignore_offsets,
      o_vec,
      w_vec,
      max_idx_vec,
      cfg,
      fixing_bag_size);
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), caller);
}

void embedding_bag_sum_template(
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& weights,
    const Tensor& per_sample_weights,
    Tensor& output,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    int64_t index_size,
    int64_t bag_num,
    int64_t vec_len,
    int64_t padding_idx,
    bool ignore_offsets) {
#define EXTEND_EMBBAG_SUM_KERNEL_VEC(vec_size) \
  EMBBAG_KERNEL_ACC(                           \
      scalar_t,                                \
      accscalar_t,                             \
      index_t,                                 \
      MODE_SUM,                                \
      vec_size,                                \
      output,                                  \
      weights,                                 \
      input,                                   \
      offset,                                  \
      offset2bag,                              \
      bag_size,                                \
      max_indices,                             \
      per_sample_weights,                      \
      index_len,                               \
      bag_num,                                 \
      vec_len,                                 \
      padding_idx,                             \
      ignore_offsets)

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weights.scalar_type(),
      "embedding_bag_sum",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_sum", [&] {
              using accscalar_t = at::acc_type<scalar_t, true>;
              int vec_size = memory::can_vectorize_up_to<scalar_t>(
                  (char*)weights.data_ptr());
              vec_size = vec_len % vec_size == 0 ? vec_size : 1;
              switch (vec_size) {
                case 8:
                  EXTEND_EMBBAG_SUM_KERNEL_VEC(8);
                  break;
                case 4:
                  EXTEND_EMBBAG_SUM_KERNEL_VEC(4);
                  break;
                case 2:
                  EXTEND_EMBBAG_SUM_KERNEL_VEC(2);
                  break;
                default:
                  EXTEND_EMBBAG_SUM_KERNEL_VEC(1);
                  break;
              };
            });
      });
#undef EXTEND_EMBBAG_SUM_KERNEL_VEC
}

void embedding_bag_mean_template(
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& weights,
    const Tensor& per_sample_weights,
    Tensor& output,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    int64_t index_size,
    int64_t bag_num,
    int64_t vec_len,
    int64_t padding_idx,
    bool ignore_offsets) {
#define EXTEND_EMBBAG_MEAN_KERNEL_VEC(vec_size) \
  EMBBAG_KERNEL_ACC(                            \
      scalar_t,                                 \
      accscalar_t,                              \
      index_t,                                  \
      MODE_MEAN,                                \
      vec_size,                                 \
      output,                                   \
      weights,                                  \
      input,                                    \
      offset,                                   \
      offset2bag,                               \
      bag_size,                                 \
      max_indices,                              \
      per_sample_weights,                       \
      index_len,                                \
      bag_num,                                  \
      vec_len,                                  \
      padding_idx,                              \
      ignore_offsets)

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weights.scalar_type(),
      "embedding_bag_mean",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_mean", [&] {
              using accscalar_t = at::acc_type<scalar_t, true>;
              int vec_size = memory::can_vectorize_up_to<scalar_t>(
                  (char*)weights.data_ptr());
              vec_size = vec_len % vec_size == 0 ? vec_size : 1;
              switch (vec_size) {
                case 8:
                  EXTEND_EMBBAG_MEAN_KERNEL_VEC(8);
                  break;
                case 4:
                  EXTEND_EMBBAG_MEAN_KERNEL_VEC(4);
                  break;
                case 2:
                  EXTEND_EMBBAG_MEAN_KERNEL_VEC(2);
                  break;
                default:
                  EXTEND_EMBBAG_MEAN_KERNEL_VEC(1);
                  break;
              };
            });
      });
#undef EXTEND_EMBBAG_MEAN_KERNEL_VEC
}

void embedding_bag_max_template(
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& weights,
    const Tensor& per_sample_weights,
    Tensor& output,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    int64_t index_size,
    int64_t bag_num,
    int64_t vec_len,
    int64_t padding_idx,
    bool ignore_offsets) {
#define EXTEND_EMBBAG_MAX_KERNEL_VEC(vec_size) \
  EMBBAG_KERNEL_NO_ACC(                        \
      scalar_t,                                \
      index_t,                                 \
      MODE_MAX,                                \
      vec_size,                                \
      output,                                  \
      weights,                                 \
      input,                                   \
      offset,                                  \
      offset2bag,                              \
      bag_size,                                \
      max_indices,                             \
      per_sample_weights,                      \
      index_len,                               \
      bag_num,                                 \
      vec_len,                                 \
      padding_idx,                             \
      ignore_offsets)

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weights.scalar_type(),
      "embedding_bag_max",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_max", [&] {
              // using accscalar_t = at::acc_type<scalar_t, true>;
              int vec_size = memory::can_vectorize_up_to<scalar_t>(
                  (char*)weights.data_ptr());
              vec_size = vec_len % vec_size == 0 ? vec_size : 1;
              switch (vec_size) {
                case 8:
                  EXTEND_EMBBAG_MAX_KERNEL_VEC(8);
                  break;
                case 4:
                  EXTEND_EMBBAG_MAX_KERNEL_VEC(4);
                  break;
                case 2:
                  EXTEND_EMBBAG_MAX_KERNEL_VEC(2);
                  break;
                default:
                  EXTEND_EMBBAG_MAX_KERNEL_VEC(1);
                  break;
              };
            });
      });
#undef EXTEND_EMBBAG_MAX_KERNEL_VEC
}

#undef EMBBAG_KERNEL_ACC
#undef EMBBAG_KERNEL_NO_ACC

/*
  The kernel EmbeddingBag is optimized for memory coleascing and thread
  efficiency. Vec design and chunk design are deployed for this kernel. In
  additional, single bag is specifically considered.(for example, when
  offset_data=0,1,2,3,4,5,...).
  Thought:
  0. Principle: One or multi chunks work for one Bag. One loop at least solves
  one bag.
  1. Implementation: Use vec<scalar_t, vec_size> to achieve higher bandwidth
  both in ATS and PVC, because it is a memory bound kernel. Use chunk design,
  chunk splitted from different WG to reach high occupancy especially when bag
  dim is much larger. The vec size is determined by device. The chunk size is
  determined by workload amounts and device resource.
  2. If it is single bag specific situation, pure copy is done for kernel.
  Single bag means offset is linear increase by 1.
  3. Passing vec size as template to kernel.

  Shortcoming:
  1. Chunk design may cause some resource waste when work items is handling
  the tail of last bag in one loop.
*/
template <typename scalar_t, typename index_t>
void EmbeddingBag_updateOutputKernel(
    const int64_t mode,
    index_t* input_data,
    index_t* offset_data,
    scalar_t* weight_data,
    scalar_t* output_data,
    index_t* offset2bag_data,
    int64_t weight_total_elem,
    int64_t input_length,
    int64_t numBags,
    int64_t weight_stride0,
    int64_t weight_stride1,
    index_t* bag_size_data,
    index_t* max_indices_data,
    scalar_t* per_sample_weights_data,
    int64_t per_sample_weights_stride,
    const bool include_last_offset,
    const index_t padding_idx,
    const bool ignore_offsets) {
  using accscalar_t = at::acc_type<scalar_t, true>;

  // vector size, query it according to machine, scalar_t and weight_data
  auto vec_size = memory::can_vectorize_up_to<scalar_t>(
      reinterpret_cast<char*>(weight_data));

  // determine per sample weights should be in calculation or not
  bool per_sample_weights_defined = per_sample_weights_data ? true : false;

  auto maxWGSize = syclMaxWorkGroupSize();

  auto gpuEuCount = syclMaxWorkItemsPerEU();

  // how many work items serve for one bag in vector sight
  auto bag_wi_num = (weight_stride0 % vec_size == 0)
      ? (weight_stride0 / vec_size)
      : (weight_stride0 / vec_size + 1);

  auto chunk_size = 32;

  // how many chunks serve for one bag
  auto bag_chunk_num = (bag_wi_num % chunk_size == 0)
      ? (bag_wi_num / chunk_size)
      : (bag_wi_num / chunk_size + 1);

  // how many work items serve for one bag in chunk sight
  bag_wi_num = bag_chunk_num * chunk_size;

  // how many chunks serve for all bag
  auto all_chunk_num = numBags * bag_chunk_num;

  // how many wi serve for all bag
  auto all_wi_num = all_chunk_num * chunk_size;

  // For huge bags number, limited wg number is set to avoid overhead of
  // groups over scheduling. WGNumber default in single tile in one time =
  // Max compute unit * 8 threads * SIMD32 per thread / max WG size * 512.
  auto WGNumber = gpuEuCount * 8 * 32 / maxWGSize * 512;

  // one or multi chunks for one bag.
  // all_wi_num <= maxWGSize: one wg is enough to finish all bags
  // bag_wi_num > (maxWGSize * WGNumber): all wg is not enough to finish one
  // bag. To avoid the inner-bag loop, all needed wg are launched
  // else: one wg is not enough to finish all bags, but all wg can finish at
  // least one bag
  auto local_range = maxWGSize;
  if (all_wi_num <= maxWGSize) {
    local_range = all_wi_num;
    WGNumber = 1;
  } else if (bag_wi_num > (maxWGSize * WGNumber)) {
    local_range = maxWGSize;
    // at least, one loop finish one bag
    WGNumber = (bag_wi_num + maxWGSize - 1) / maxWGSize;
  } else {
    for (auto factor = 0; (((maxWGSize - factor * 8) >= 8)); ++factor) {
      auto infactor = maxWGSize - factor * 8;
      if (all_wi_num % infactor == 0) {
        if ((all_wi_num / infactor) > WGNumber) {
          local_range = infactor;
        } else {
          WGNumber = all_wi_num / infactor;
          local_range = infactor;
        }
        break;
      }
    }
  }

  // for outer bag loop, how many bag finish in one loop
  auto bagsPerLoop = WGNumber * local_range / chunk_size / bag_chunk_num;

  // total work item size
  auto global_range = WGNumber * local_range;

  bool if_align_vector = ((weight_stride0 % 2 == 0) || (sizeof(scalar_t) != 2));

// launch vec kernel for embeddingbag, code pass according to vec size
#define VEC_EMBBAG_KERNEL(vec_size)                                         \
  {                                                                         \
    auto input = input_data;                                                \
    auto offset = offset_data;                                              \
    auto weight = weight_data;                                              \
    auto output = output_data;                                              \
    auto offset2bag = offset2bag_data;                                      \
    auto bag_size = bag_size_data;                                          \
    auto per_sample_weights =                                               \
        per_sample_weights_defined ? per_sample_weights_data : weight_data; \
    auto max_indices = mode == MODE_MAX ? max_indices_data : nullptr;       \
    using vec_t = memory::aligned_vector<scalar_t, vec_size>;               \
    auto caller = EmbeddingBagUpdateOutputKernelFunctor<                    \
        vec_size,                                                           \
        vec_t,                                                              \
        scalar_t,                                                           \
        accscalar_t,                                                        \
        index_t>(                                                           \
        mode,                                                               \
        input,                                                              \
        offset,                                                             \
        weight,                                                             \
        output,                                                             \
        offset2bag,                                                         \
        bag_size,                                                           \
        per_sample_weights_defined,                                         \
        per_sample_weights,                                                 \
        per_sample_weights_stride,                                          \
        max_indices,                                                        \
        WGNumber,                                                           \
        numBags,                                                            \
        weight_total_elem,                                                  \
        chunk_size,                                                         \
        bag_chunk_num,                                                      \
        bag_wi_num,                                                         \
        bagsPerLoop,                                                        \
        input_length,                                                       \
        weight_stride0,                                                     \
        weight_stride1,                                                     \
        include_last_offset,                                                \
        padding_idx,                                                        \
        if_align_vector);                                                   \
    sycl_kernel_submit(                                                     \
        global_range, local_range, getCurrentSYCLQueue(), caller);          \
  };

  switch (vec_size) {
    case 16: {
      VEC_EMBBAG_KERNEL(16);
      break;
    }
    case 8: {
      VEC_EMBBAG_KERNEL(8);
      break;
    }
    case 4: {
      VEC_EMBBAG_KERNEL(4);
      break;
    }
    case 2: {
      VEC_EMBBAG_KERNEL(2);
      break;
    }
    case 1: {
      VEC_EMBBAG_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for EmbeddingBag. vec size ",
          vec_size);
  }
#undef VEC_EMBBAG_KERNEL
}

template <typename scalar_t, typename index_t>
Tensor embedding_bag_backward_dpcpp_sum_avg(
    const Tensor& grad,
    const Tensor& indices_t,
    const Tensor& offset2bag_t,
    const Tensor& bag_size_t,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights_t,
    int64_t padding_idx) {
  auto indices = indices_t.contiguous();
  auto offset2bag = offset2bag_t.contiguous();
  auto bag_size = bag_size_t.contiguous();
  auto per_sample_weights = per_sample_weights_t.contiguous();

  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  ptrdiff_t numel = indices.numel();

  if (numel == 0) {
    // return empty bags
    return at::zeros({num_weights, grad.size(1)}, grad.options());
  }

  // int64_t stride = grad_weight.stride(0);

  auto sorted_indices = at::empty_like(indices);
  auto sorted_begin = sorted_indices.data_ptr<index_t>();
  auto orig_indices = at::empty_like(indices);
  auto orig_begin = orig_indices.data_ptr<index_t>();

  // directly
  {
    sorted_indices.copy_(indices);
    pstl::itoa(orig_begin, orig_begin + numel, (index_t)0);
    pstl::sort<index_t, index_t>(
        indices.data_ptr<index_t>(), sorted_begin, orig_begin, numel, false);
  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = at::empty_like(sorted_indices);
    index_t* count_begin = count.data_ptr<index_t>();
    // Take the maximum of each count per unique key:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    //
    embedding_bag_backward_dpcpp_sum_avg_functor<index_t> f;
    pstl::count_by_segment<index_t, index_t, index_t>(
        sorted_begin, sorted_begin + numel, count_begin, f);
  }

  return embedding_backward_deterministic_kernel<scalar_t, index_t>(
      grad,
      orig_indices,
      sorted_indices,
      count,
      num_weights,
      padding_idx,
      mode == MODE_MEAN,
      offset2bag,
      bag_size,
      per_sample_weights);
}

template <typename scalar_t, typename index_t>
struct EmbeddingBagAccGradParametersKernelMaxFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto max_indices_ptr = max_indices_data;
    auto gradOutput_ptr = gradOutput_data;
    auto gradWeight_ptr = gradWeight_data;

    auto chunkOffset = item.get_group()[0] * item.get_local_range()[1] +
        item.get_local_id()[1];

    for (auto chunk = chunkOffset; chunk < numChunks;
         chunk += item.get_group_range()[0] * item.get_global_range()[1]) {
      auto featureDim = (chunk % chunksPerBag) * item.get_local_range(0) +
          item.get_local_id(0);
      if (featureDim < stride) {
        auto bag = chunk / chunksPerBag;

        auto word_idx = max_indices_ptr[bag * stride + featureDim];
        if (word_idx >= 0) {
          // If bag is empty, we have max_indices[idx] set to -1 in forward.
          atomicAdd(
              (sycl_global_ptr<scalar_t>)&(
                  gradWeight_ptr[word_idx * stride + featureDim]),
              gradOutput_ptr[bag * stride + featureDim]);
        }
      }
    }
  }
  EmbeddingBagAccGradParametersKernelMaxFunctor(
      index_t* max_indices_data_,
      scalar_t* gradOutput_data_,
      scalar_t* gradWeight_data_,
      int64_t stride_,
      int64_t chunksPerBag_,
      int64_t numChunks_)
      : max_indices_data(max_indices_data_),
        gradOutput_data(gradOutput_data_),
        gradWeight_data(gradWeight_data_),
        stride(stride_),
        chunksPerBag(chunksPerBag_),
        numChunks(numChunks_) {}

 private:
  index_t* max_indices_data;
  scalar_t* gradOutput_data;
  scalar_t* gradWeight_data;
  int64_t stride;
  int64_t chunksPerBag;
  int64_t numChunks;
};

template <typename scalar_t, typename index_t>
void EmbeddingBag_accGradParametersKernel_max(
    index_t* max_indices,
    scalar_t* gradOutput,
    scalar_t* gradWeight,
    int64_t stride,
    int64_t numBags) {
  auto chunksPerBag = CeilDiv(stride, (int64_t)64);
  auto numChunks = numBags * chunksPerBag;
  auto kernel_range = 1024 * 64;

  auto max_indices_data = max_indices;
  auto gradOutput_data = gradOutput;
  auto gradWeight_data = gradWeight;

  auto caller =
      EmbeddingBagAccGradParametersKernelMaxFunctor<scalar_t, index_t>(
          max_indices_data,
          gradOutput_data,
          gradWeight_data,
          stride,
          chunksPerBag,
          numChunks);

  auto global_range = sycl::range<2>(kernel_range, 4);
  auto local_range = sycl::range<2>(64, 4);
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), caller);
}

template <typename scalar_t, typename index_t>
Tensor embedding_bag_backward_dpcpp_max(
    const Tensor& grad,
    const Tensor& max_indices_t,
    int64_t num_weights,
    int64_t padding_idx) {
  auto max_indices = max_indices_t.contiguous();
  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());
  int64_t stride = grad_weight.stride(0);
  int64_t numBags = grad.size(0);

  EmbeddingBag_accGradParametersKernel_max<scalar_t>(
      max_indices.data_ptr<index_t>(),
      grad.data_ptr<scalar_t>(),
      grad_weight.data_ptr<scalar_t>(),
      stride,
      numBags);

  return grad_weight;
}

template <typename scalar_t, typename index_t>
static void _embedding_bag_per_sample_weights_backward_kernel(
    const scalar_t* grad,
    int64_t grad_stride0,
    int64_t grad_stride1,
    const scalar_t* weight,
    int64_t weight_stride0,
    int64_t weight_stride1,
    const index_t* indices, // contiguous
    const index_t* offset2bag, // contiguous
    int64_t num_samples,
    int64_t embedding_features,
    scalar_t* output,
    index_t padding_idx) {
  using accscalar_t = at::acc_type<scalar_t, true>;

  int64_t max_group_size = 64;

  int64_t num_group = (num_samples + max_group_size - 1) / max_group_size;
  auto global_range{num_group * max_group_size};
  auto local_range{max_group_size};

  auto caller = EmbeddingBagPerSampleWeightsBackwardKernelFunctor<
      scalar_t,
      index_t,
      accscalar_t>(
      grad,
      grad_stride0,
      grad_stride1,
      weight,
      weight_stride0,
      weight_stride1,
      indices,
      offset2bag,
      num_samples,
      embedding_features,
      output,
      padding_idx,
      num_group,
      max_group_size);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), caller);
}

} // namespace detail

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_dpcpp(
    const Tensor& weight_t,
    const Tensor& indices_t,
    const Tensor& offsets_t,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights_t,
    bool include_last_offset,
    int64_t padding_idx) {
  TORCH_CHECK(
      indices_t.dim() == 1 || indices_t.dim() == 2,
      "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",
      indices_t.dim());
  if (indices_t.dim() == 1) {
    TORCH_CHECK(
        offsets_t.dim() == 1,
        "offsets has to be a 1D Tensor, but got Tensor of dimension ",
        offsets_t.dim());
  }
  TORCH_CHECK(
      weight_t.dim() == 2,
      "weight has to be a 2D Tensor, but got Tensor of dimension ",
      weight_t.dim());

  auto weight = weight_t.contiguous();
  auto indices_original = indices_t.contiguous();
  auto offsets_original = offsets_t.contiguous();
  auto per_sample_weights = per_sample_weights_t.contiguous();

  Tensor indices, offsets;
  std::tie(indices, offsets) =
      promoteIndicesAndOffsets(indices_original, offsets_original);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag_dpcpp", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag_dpcpp", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag_dpcpp", indices_arg, offsets_arg);
  checkSameGPU("embedding_bag_dpcpp", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkSameGPU("embedding_bag_dpcpp", weight_arg, indices_arg);
  checkSameGPU("embedding_bag_dpcpp", weight_arg, offsets_arg);

  bool ignore_offsets = indices.sizes().size() == 2;
  int64_t numIndices = indices.numel();
  int64_t numBags = ignore_offsets ? indices.size(0) : offsets.size(0);

  // include last offset = True, means the last element of offsets will be set
  // equal to the length of input. Default it is False.
  if (include_last_offset) {
    TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
    numBags -= 1;
  }

  auto bag_size = at::empty(numBags, indices.options());
  auto offset2bag = at::empty({indices.size(0)}, indices.options());
  auto output = at::empty({numBags, weight.size(1)}, weight.options());

  Tensor max_indices = at::empty({numBags, weight.size(1)}, indices.options());

#ifndef VEC_EMBBAG_KERNEL_OPT
#define EXTEND_EMBBAG_TEMPLATE(mode) \
  embedding_bag_##mode##_template(   \
      indices,                       \
      offsets,                       \
      weight,                        \
      per_sample_weights,            \
      output,                        \
      offset2bag,                    \
      bag_size,                      \
      max_indices,                   \
      numIndices,                    \
      numBags,                       \
      weight.stride(0),              \
      padding_idx,                   \
      ignore_offsets)

  switch (mode) {
    case MODE_SUM:
      EXTEND_EMBBAG_TEMPLATE(sum);
      break;
    case MODE_MEAN:
      EXTEND_EMBBAG_TEMPLATE(mean);
      break;
    case MODE_MAX:
      EXTEND_EMBBAG_TEMPLATE(max);
      break;
    default:
      TORCH_CHECK(0, "Invalid EmbeddingBag mode (max, sum, mean) ...");
  };
#undef EXTEND_EMBBAG_TEMPLATE
#else
  int64_t weight_total_elem = weight.numel();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "embedding_bag_dpcpp",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_dpcpp", [&] {
              EmbeddingBag_updateOutputKernel<scalar_t, index_t>(
                  mode,
                  indices.data_ptr<index_t>(),
                  offsets.data_ptr<index_t>(),
                  weight.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>(),
                  offset2bag.data_ptr<index_t>(),
                  weight_total_elem,
                  numIndices,
                  numBags,
                  weight.stride(0),
                  weight.stride(1),
                  bag_size.data_ptr<index_t>(),
                  mode == MODE_MAX ? max_indices.data_ptr<index_t>() : NULL,
                  per_sample_weights.defined()
                      ? per_sample_weights.data_ptr<scalar_t>()
                      : NULL,
                  per_sample_weights.defined() ? per_sample_weights.stride(0)
                                               : 0,
                  include_last_offset,
                  padding_idx,
                  ignore_offsets);
            });
      });
#endif

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_dpcpp(
    const Tensor& grad_t,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights,
    int64_t padding_idx) {
  Tensor grad = grad_t.contiguous();
  Tensor result;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_dense_backward_dpcpp",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_dense_backward_dpcpp", [&] {
              switch (mode) {
                case MODE_SUM:
                case MODE_MEAN:
                  if (mode == MODE_MEAN) {
                    TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
                  }
                  result =
                      embedding_bag_backward_dpcpp_sum_avg<scalar_t, index_t>(
                          grad,
                          indices,
                          offset2bag,
                          bag_size,
                          num_weights,
                          scale_grad_by_freq,
                          mode,
                          per_sample_weights,
                          padding_idx);
                  return result;
                case MODE_MAX:
                  TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
                  result = embedding_bag_backward_dpcpp_max<scalar_t, index_t>(
                      grad, max_indices, num_weights, padding_idx);
                  return result;
                default:
                  TORCH_CHECK(
                      0,
                      "Unknown mode for embedding_bag_backward_dpcpp ",
                      mode);
              }
            });
      });
  return result;
}

Tensor _embedding_bag_per_sample_weights_backward_dpcpp(
    const Tensor& grad,
    const Tensor& weight, // NB: embedding table, not per_sample_weights
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.size(1);

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.size(1) == embedding_features);

  auto output = at::empty({num_samples}, grad.options());

  // Early return when there is no samples in the batch. This saves unnecesary
  // kernel launch
  if (num_samples == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "_embedding_bag_per_sample_weights_backward_dpcpp",
      [&]() {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(),
            "_embedding_bag_per_sample_weights_backward_dpcpp",
            [&]() {
              _embedding_bag_per_sample_weights_backward_kernel<
                  scalar_t,
                  index_t>(
                  grad.data_ptr<scalar_t>(),
                  grad.stride(0),
                  grad.stride(1),
                  weight.data_ptr<scalar_t>(),
                  weight.stride(0),
                  weight.stride(1),
                  indices.data_ptr<index_t>(),
                  offset2bag.data_ptr<index_t>(),
                  num_samples,
                  embedding_features,
                  output.data_ptr<scalar_t>(),
                  padding_idx);
            });
      });
  return output;
}

} // namespace at::native::xpu
