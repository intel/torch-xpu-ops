/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ops/_embedding_bag_forward_only_native.h>
#include <ATen/ops/_embedding_bag_native.h>

#include <ATen/native/xpu/sycl/EmbeddingBagKernels.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_xpu(
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

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only_xpu(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  return _embedding_bag_xpu(
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

Tensor _embedding_bag_dense_backward_xpu(
    const Tensor& grad,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const std::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  return native::xpu::_embedding_bag_dense_backward_kernel(
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

Tensor _embedding_bag_per_sample_weights_backward_xpu(
    const Tensor& grad,
    const Tensor& weight, // NB: embedding table, not per_sample_weights
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
  return native::xpu::_embedding_bag_per_sample_weights_backward_kernel(
      grad, weight, indices_, offsets_, offset2bag, mode, padding_idx);
}
} // namespace native
} // namespace at
