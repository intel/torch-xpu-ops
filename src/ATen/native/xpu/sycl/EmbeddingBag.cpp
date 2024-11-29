#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/EmbeddingBackwardKernel.h>
#include <ATen/native/xpu/sycl/EmbeddingBag.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>

#include <ATen/native/xpu/sycl/EmbeddingBagKernels.h>

namespace at::native::xpu {

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

template <
    typename scalar_t,
    typename accscalar_t,
    typename index_t,
    int mode,
    int vec_size>
void embedding_bag(
    scalar_t* const output,
    const scalar_t* const weights,
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
    int64_t num_row) {
  using vec_t = at::detail::Array<scalar_t, vec_size>;
  using vec_acc_t = at::detail::Array<accscalar_t, vec_size>;
  using vec_idx_t = at::detail::Array<index_t, vec_size>;
  using KernelClass = EmbeddingBagKernelFunctor<
      scalar_t,
      accscalar_t,
      index_t,
      mode,
      vec_size,
      vec_t,
      vec_acc_t,
      vec_idx_t>;

  vec_t* o_vec = reinterpret_cast<vec_t*>(output);
  const vec_t* w_vec = reinterpret_cast<const vec_t*>(weights);
  vec_idx_t* max_idx_vec = reinterpret_cast<vec_idx_t*>(max_index);

  vec_len = vec_len / vec_size;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      bag_num, vec_len, 1, bag_num, true, BatchKernelConfig::Policy::pAdaptive);

  index_t fixing_bag_size = ignore_offsets ? index_size / bag_num : 0;
  auto kfn = KernelClass(
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
      fixing_bag_size,
      num_row);
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
}

#define EMBBAG_KERNEL_ACC(                                       \
    scalar_t,                                                    \
    accscalar_t,                                                 \
    index_t,                                                     \
    mode,                                                        \
    vec_size,                                                    \
    output,                                                      \
    weight,                                                      \
    input,                                                       \
    offset,                                                      \
    offset2bag,                                                  \
    bag_size,                                                    \
    max_indices,                                                 \
    per_sample_weights,                                          \
    index_len,                                                   \
    bag_num,                                                     \
    vec_len,                                                     \
    padding_idx,                                                 \
    ignore_offsets,                                              \
    num_row)                                                     \
  embedding_bag<scalar_t, accscalar_t, index_t, mode, vec_size>( \
      output.mutable_data_ptr<scalar_t>(),                       \
      weight.const_data_ptr<scalar_t>(),                         \
      indices.const_data_ptr<index_t>(),                         \
      offsets.const_data_ptr<index_t>(),                         \
      offset2bag.mutable_data_ptr<index_t>(),                    \
      bag_size.mutable_data_ptr<index_t>(),                      \
      max_indices.mutable_data_ptr<index_t>(),                   \
      per_sample_weights.defined()                               \
          ? per_sample_weights.const_data_ptr<scalar_t>()        \
          : nullptr,                                             \
      index_size,                                                \
      bag_num,                                                   \
      vec_len,                                                   \
      padding_idx,                                               \
      ignore_offsets,                                            \
      num_row)

#define EMBBAG_KERNEL_NO_ACC(                                 \
    scalar_t,                                                 \
    index_t,                                                  \
    mode,                                                     \
    vec_size,                                                 \
    output,                                                   \
    weight,                                                   \
    input,                                                    \
    offset,                                                   \
    offset2bag,                                               \
    bag_size,                                                 \
    max_indices,                                              \
    per_sample_weights,                                       \
    index_len,                                                \
    bag_num,                                                  \
    vec_len,                                                  \
    padding_idx,                                              \
    ignore_offsets,                                           \
    num_row)                                                  \
  embedding_bag<scalar_t, scalar_t, index_t, mode, vec_size>( \
      output.mutable_data_ptr<scalar_t>(),                    \
      weight.const_data_ptr<scalar_t>(),                      \
      indices.const_data_ptr<index_t>(),                      \
      offsets.const_data_ptr<index_t>(),                      \
      offset2bag.mutable_data_ptr<index_t>(),                 \
      bag_size.mutable_data_ptr<index_t>(),                   \
      max_indices.mutable_data_ptr<index_t>(),                \
      per_sample_weights.defined()                            \
          ? per_sample_weights.const_data_ptr<scalar_t>()     \
          : nullptr,                                          \
      index_size,                                             \
      bag_num,                                                \
      vec_len,                                                \
      padding_idx,                                            \
      ignore_offsets,                                         \
      num_row)

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
  uint64_t num_row = weights.size(0);
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
      ignore_offsets,                          \
      num_row)

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weights.scalar_type(),
      "embedding_bag_sum_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_sum_xpu", [&] {
              using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
              int vec_size = memory::can_vectorize_up_to<scalar_t>(
                  (char*)weights.const_data_ptr());
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
  uint64_t num_row = weights.size(0);
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
      ignore_offsets,                           \
      num_row)

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weights.scalar_type(),
      "embedding_bag_mean_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_mean_xpu", [&] {
              using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
              int vec_size = memory::can_vectorize_up_to<scalar_t>(
                  (char*)weights.const_data_ptr());
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
  uint64_t num_row = weights.size(0);
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
      ignore_offsets,                          \
      num_row)

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weights.scalar_type(),
      "embedding_bag_max_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_max_xpu", [&] {
              // using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
              int vec_size = memory::can_vectorize_up_to<scalar_t>(
                  (char*)weights.const_data_ptr());
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

template <typename scalar_t, typename index_t>
Tensor embedding_bag_backward_xpu_sum_avg(
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
  auto sorted_begin = sorted_indices.mutable_data_ptr<index_t>();
  auto orig_indices = at::empty_like(indices);
  auto orig_begin = orig_indices.mutable_data_ptr<index_t>();

  // directly
  {
    sorted_indices.copy_(indices);
    pstl::itoa(orig_begin, orig_begin + numel, (index_t)0);
    pstl::sort<index_t, index_t>(
        indices.const_data_ptr<index_t>(),
        sorted_begin,
        orig_begin,
        numel,
        false);
  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = at::empty_like(sorted_indices);
    index_t* count_begin = count.mutable_data_ptr<index_t>();
    // Take the maximum of each count per unique key:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    //
    EmbeddingBagBackwardSumAvgFunctor<index_t> f;
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
    auto max_indices_ptr = max_indices_data_;
    auto gradOutput_ptr = gradOutput_data_;
    auto gradWeight_ptr = gradWeight_data_;

    auto chunkOffset = item.get_group()[0] * item.get_local_range()[1] +
        item.get_local_id()[1];

    for (auto chunk = chunkOffset; chunk < numChunks_;
         chunk += item.get_group_range()[0] * item.get_global_range()[1]) {
      auto featureDim = (chunk % chunksPerBag_) * item.get_local_range(0) +
          item.get_local_id(0);
      if (featureDim < stride_) {
        auto bag = chunk / chunksPerBag_;

        auto word_idx = max_indices_ptr[bag * stride_ + featureDim];
        if (word_idx >= 0) {
          // If bag is empty, we have max_indices[idx] set to -1 in forward.
          atomicAdd(
              (sycl_global_ptr<
                  scalar_t>)(&gradWeight_ptr[word_idx * stride_ + featureDim]),
              gradOutput_ptr[bag * stride_ + featureDim]);
        }
      }
    }
  }
  EmbeddingBagAccGradParametersKernelMaxFunctor(
      const index_t* max_indices_data,
      const scalar_t* gradOutput_data,
      scalar_t* gradWeight_data,
      int64_t stride,
      int64_t chunksPerBag,
      int64_t numChunks)
      : max_indices_data_(max_indices_data),
        gradOutput_data_(gradOutput_data),
        gradWeight_data_(gradWeight_data),
        stride_(stride),
        chunksPerBag_(chunksPerBag),
        numChunks_(numChunks) {}

 private:
  const index_t* max_indices_data_;
  const scalar_t* gradOutput_data_;
  scalar_t* gradWeight_data_;
  int64_t stride_;
  int64_t chunksPerBag_;
  int64_t numChunks_;
};

template <typename scalar_t, typename index_t>
void EmbeddingBag_accGradParametersKernel_max(
    const index_t* max_indices,
    const scalar_t* gradOutput,
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
Tensor embedding_bag_backward_xpu_max(
    const Tensor& grad,
    const Tensor& max_indices_t,
    int64_t num_weights,
    int64_t padding_idx) {
  globalContext().alertNotDeterministic("embedding_bag_backward_xpu_max");

  auto max_indices = max_indices_t.contiguous();
  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());
  int64_t stride = grad_weight.stride(0);
  int64_t numBags = grad.size(0);

  EmbeddingBag_accGradParametersKernel_max<scalar_t>(
      max_indices.const_data_ptr<index_t>(),
      grad.const_data_ptr<scalar_t>(),
      grad_weight.mutable_data_ptr<scalar_t>(),
      stride,
      numBags);

  return grad_weight;
}

template <typename scalar_t, typename index_t>
void _embedding_bag_per_sample_weights_backward_impl(
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

  using Kernel = EmbeddingBagPerSampleWeightsBackwardKernelFunctor<
      scalar_t,
      index_t,
      accscalar_t>;

  int64_t max_group_size = syclMaxWorkGroupSize<Kernel>();

  int64_t num_group = (num_samples + max_group_size - 1) / max_group_size;
  auto global_range{num_group * max_group_size};
  auto local_range{max_group_size};

  auto caller = Kernel(
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

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_kernel(
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
  checkScalarTypes("embedding_bag_kernel", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag_kernel", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag_kernel", indices_arg, offsets_arg);
  checkSameGPU("embedding_bag_kernel", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  checkSameGPU("embedding_bag_kernel", weight_arg, indices_arg);
  checkSameGPU("embedding_bag_kernel", weight_arg, offsets_arg);

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

  auto bag_size = at::empty(offsets.sizes(), indices.options());
  auto offset2bag = at::empty({indices.size(0)}, indices.options());
  auto output = at::empty({numBags, weight.size(1)}, weight.options());

  Tensor max_indices;

  if (mode == MODE_MAX) {
    max_indices = at::empty({numBags, weight.size(1)}, indices.options());
  } else {
    // No need to allocate if we aren't doing a backwards pass
    max_indices = at::empty({0}, indices.options());
  }

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

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_kernel(
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
  auto indices_arg = TensorArg(indices, "indices", 1);
  auto grad_arg = TensorArg(grad, "grad", 1);
  checkSameGPU("embedding_bag_cuda", grad_arg, indices_arg);

  Tensor result;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_dense_backward_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_dense_backward_xpu", [&] {
              switch (mode) {
                case MODE_SUM:
                case MODE_MEAN:
                  if (mode == MODE_MEAN) {
                    TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
                  }
                  result =
                      embedding_bag_backward_xpu_sum_avg<scalar_t, index_t>(
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
                  result = embedding_bag_backward_xpu_max<scalar_t, index_t>(
                      grad, max_indices, num_weights, padding_idx);
                  return result;
                default:
                  TORCH_CHECK(
                      0, "Unknown mode for embedding_bag_backward_xpu ", mode);
              }
            });
      });
  return result;
}

Tensor _embedding_bag_per_sample_weights_backward_kernel(
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
      "_embedding_bag_per_sample_weights_backward_xpu",
      [&]() {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(),
            "_embedding_bag_per_sample_weights_backward_xpu",
            [&]() {
              _embedding_bag_per_sample_weights_backward_impl<
                  scalar_t,
                  index_t>(
                  grad.const_data_ptr<scalar_t>(),
                  grad.stride(0),
                  grad.stride(1),
                  weight.const_data_ptr<scalar_t>(),
                  weight.stride(0),
                  weight.stride(1),
                  indices.const_data_ptr<index_t>(),
                  offset2bag.const_data_ptr<index_t>(),
                  num_samples,
                  embedding_features,
                  output.mutable_data_ptr<scalar_t>(),
                  padding_idx);
            });
      });
  return output;
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
