#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/autograd.h>
#include "sdp_utils.h"

using namespace torch::autograd;

namespace at {

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_native_impl(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    const c10::optional<Tensor>& dropout_mask,
    c10::optional<double> scale) {
  if (query_.is_nested() || key.is_nested() || value.is_nested()) {
    TORCH_CHECK(
        query_.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
        "scaled_dot_product_attention: If inputs are nested tensors they must be contiguous");
  }
  auto attn_mask = attn_mask_;
  // Naive, composite implementation defined here.

  // Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for
  // math
  bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
  const auto scaling_factor =
      sdp::native_calculate_scale(
          query_, is_negative_scaling ? std::abs(scale.value()) : scale)
          .sqrt();

  const auto query = query_ *
      (is_negative_scaling ? c10::SymFloat(0.0) - scaling_factor
                           : scaling_factor);
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");

    // Replace attn_mask with causal mask; lower triangular elements take part
    // in attention.
    const auto L = query.sym_size(-2), S = key.sym_size(-2);
    attn_mask =
        at::ones_symint({L, S}, query.options().dtype(at::kBool)).tril();
    attn_mask = sdp::convert_boolean_attn_mask(attn_mask, query.dtype());
  }
  auto attn = at::matmul(query, key.transpose(-2, -1) * scaling_factor);
  if (attn_mask.has_value()) {
    if (at::areAnyTensorSubclassLike({attn, *attn_mask})) {
      attn = attn.add(*attn_mask);
    } else {
      attn.add_(*attn_mask);
    }
  }
  attn = at::softmax(attn, -1);
  if (dropout_p > 0.0) {
    if (dropout_mask.has_value()) {
      // In order to validate the correctness of the fused kernels, we need to
      // use the same dropout mask in order to compare the results.
      TORCH_WARN_ONCE("Dropout mask should only be used for testing purposes.");
      attn = attn.masked_fill(dropout_mask->logical_not(), 0.0);
      auto dropout_scaling = 1.0 / (1 - dropout_p);
      return std::make_tuple(at::matmul(attn, value * dropout_scaling), attn);
    } else {
      attn = at::dropout(attn, dropout_p, true);
    }
  }

  return std::make_tuple(at::matmul(attn, value), attn);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::
    _scaled_dot_product_attention_math(
        const Tensor& query_,
        const Tensor& key,
        const Tensor& value,
        const c10::optional<Tensor>& attn_mask_,
        double dropout_p,
        bool is_causal,
        const c10::optional<Tensor>& dropout_mask,
        c10::optional<double> scale) {
  // on ATSM, the efficient_attention path is not available
  // With naive math path, oneDNN matmul has overflow issue with fp16 inputs
  // as a WA, convert fp16 inputs to fp32
  if (query_.requires_grad() || key.requires_grad() || value.requires_grad()) {
    return AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        query_.scalar_type(),
        "scaled_dot_product_attention_math",
        [&] {
          bool is_half = std::is_same<scalar_t, at::Half>::value;
          if (is_half) {
            c10::optional<Tensor> attn_mask_fp32;
            Tensor query_fp32 = query_.to(at::kFloat);
            Tensor key_fp32 = key.to(at::kFloat);
            Tensor value_fp32 = value.to(at::kFloat);
            if (attn_mask_.has_value()) {
              attn_mask_fp32 = attn_mask_.value().to(at::kFloat);
            } else {
              attn_mask_fp32 = attn_mask_;
            }
            auto [attn_output, attn_weight] =
                _scaled_dot_product_attention_math_native_impl(
                    query_fp32,
                    key_fp32,
                    value_fp32,
                    attn_mask_fp32,
                    dropout_p,
                    is_causal,
                    dropout_mask,
                    scale);
            return std::make_tuple(
                attn_output.to(at::kHalf), attn_weight.to(at::kHalf));
          }
          return _scaled_dot_product_attention_math_native_impl(
              query_,
              key,
              value,
              attn_mask_,
              dropout_p,
              is_causal,
              dropout_mask,
              scale);
        });
  } else {
    // accelerate for inference
    return AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        query_.scalar_type(),
        "scaled_dot_product_attention_math",
        [&] {
          bool is_half = std::is_same<scalar_t, at::Half>::value;
          if (is_half) {
            c10::optional<Tensor> attn_mask_fp32;
            Tensor query_fp32 = query_.to(at::kFloat);
            Tensor key_fp32 = key.to(at::kFloat);
            Tensor value_fp32 = value.to(at::kFloat);
            if (attn_mask_.has_value()) {
              attn_mask_fp32 = attn_mask_.value().to(at::kFloat);
            } else {
              attn_mask_fp32 = attn_mask_;
            }
            auto [attn_output, attn_weight] =
                _scaled_dot_product_attention_math_native_impl(
                    query_fp32,
                    key_fp32,
                    value_fp32,
                    attn_mask_fp32,
                    dropout_p,
                    is_causal,
                    dropout_mask,
                    scale);
            return std::make_tuple(
                attn_output.to(at::kHalf), attn_weight.to(at::kHalf));
          }
          return _scaled_dot_product_attention_math_native_impl(
              query_,
              key,
              value,
              attn_mask_,
              dropout_p,
              is_causal,
              dropout_mask,
              scale);
        });
  }
}

} // namespace at