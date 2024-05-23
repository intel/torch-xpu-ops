#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>

#include <aten/sycl/EmbeddingKernels.h>

namespace at {

std::tuple<Tensor, Tensor, Tensor, Tensor> XPUNativeFunctions::_embedding_bag(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  return native::xpu::_embedding_bag_dpcpp(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset,
      padding_idx);
}

} // namespace at
