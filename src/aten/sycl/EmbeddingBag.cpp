#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>

#include <aten/sycl/EmbeddingBagKernel.h>
#include <aten/sycl/MemoryAccess.h>

using namespace at::native::xpu::detail;

namespace at::native::xpu {

namespace detail {

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

} // namespace at::native::xpu
