#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/EmbeddingBag.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>

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
  vec_t* w_vec = reinterpret_cast<vec_t*>(weights);
  vec_idx_t* max_idx_vec = reinterpret_cast<vec_idx_t*>(max_index);

  vec_len = vec_len / vec_size;
  BatchKernelConfig cfg = {
      bag_num, vec_len, 1, bag_num, true, BatchKernelConfig::Policy::pAdaptive};
  cfg.template build<KernelClass>();

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
      fixing_bag_size);
  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
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
  embedding_bag<scalar_t, accscalar_t, index_t, mode, vec_size>(             \
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
  embedding_bag<scalar_t, scalar_t, index_t, mode, vec_size>(                \
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
      "embedding_bag_sum_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_sum_xpu", [&] {
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
      "embedding_bag_mean_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_mean_xpu", [&] {
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
      "embedding_bag_max_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_max_xpu", [&] {
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

  auto bag_size = at::empty(numBags, indices.options());
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

} // namespace at::native::xpu
