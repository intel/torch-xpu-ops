#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>

#include <aten/sycl/EmbeddingBagKernel.h>

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

Tensor XPUNativeFunctions::_embedding_bag_dense_backward(
    const Tensor& grad,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  return native::xpu::_embedding_bag_dense_backward_dpcpp(
      grad,
      indices,
      offset2bag,
      bag_size,
      maximum_indices,
      num_weights,
      scale_grad_by_freq,
      mode,
      per_sample_weights,
      padding_idx);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> XPUNativeFunctions::
    _embedding_bag_forward_only(
        const Tensor& weight,
        const Tensor& indices,
        const Tensor& offsets,
        bool scale_grad_by_freq,
        int64_t mode,
        bool sparse,
        const c10::optional<Tensor>& per_sample_weights_opt,
        bool include_last_offset,
        int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
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

Tensor XPUNativeFunctions::_embedding_bag_per_sample_weights_backward(
    const Tensor& grad,
    const Tensor& weight, // NB: embedding table, not per_sample_weights
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
  return native::xpu::_embedding_bag_per_sample_weights_backward_dpcpp(
      grad, weight, indices_, offsets_, offset2bag, mode, padding_idx);
}

} // namespace at
