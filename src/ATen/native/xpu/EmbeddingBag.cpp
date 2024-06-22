#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/EmbeddingBagKernels.h>

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
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",
      indices.dim());
  if (indices.dim() == 1) {
    TORCH_CHECK(
        offsets.dim() == 1,
        "offsets has to be a 1D Tensor, but got Tensor of dimension ",
        offsets.dim());
  }
  TORCH_CHECK(
      weight.dim() == 2,
      "weight has to be a 2D Tensor, but got Tensor of dimension ",
      weight.dim());

  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  return native::xpu::_embedding_bag_kernel(
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
  return _embedding_bag(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights_opt,
      include_last_offset,
      padding_idx);
}
} // namespace at
