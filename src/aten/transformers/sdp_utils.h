#pragma once
#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/SymFloat.h>

using namespace at;
namespace sdp {

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : c10::SymFloat(query.sym_size(-1)).sqrt();
  return c10::SymFloat(softmax_scale);
}

inline c10::SymFloat native_calculate_scale(
    const at::Tensor& query,
    c10::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

inline c10::optional<at::Tensor> convert_boolean_attn_mask(
    const c10::optional<at::Tensor>& attn_mask,
    caffe2::TypeMeta dtype) {
  // Pass through
  if (!attn_mask.has_value()) {
    return c10::nullopt;
  }
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask->dtype() == at::kBool) {
    auto new_attn_mask = at::zeros_like(attn_mask.value(), dtype);
    // TODO Use the max type of the input and output
    new_attn_mask.masked_fill_(
        attn_mask->logical_not(), -std::numeric_limits<double>::infinity());
    return new_attn_mask;
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}
} // namespace sdp