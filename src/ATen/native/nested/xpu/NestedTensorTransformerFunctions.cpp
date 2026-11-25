/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/nested/xpu/sycl/NestedTensorTransformerFunctionKernels.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_native.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_native.h>

namespace at::native {

namespace {

int64_t padded_tensor_numel(const Tensor& sizes) {
  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_row_length = sizes.sizes()[1];
  const auto* sizes_data = sizes.const_data_ptr<int64_t>();
  int64_t numel = 0;
  for (const auto row_num : c10::irange(sizes_num_rows)) {
    const auto* row_ptr = sizes_data + row_num * sizes_row_length;
    int64_t prod = 1;
    for (const auto idx : c10::irange(sizes_row_length)) {
      prod *= row_ptr[idx];
    }
    numel += prod;
  }
  return numel;
}

} // namespace

Tensor nested_from_padded_xpu(
    const Tensor& padded,
    const Tensor& sizes,
    bool do_transform_0213) {
  if (padded.dim() > 1 && padded.dim() < 5) {
    // Instead of erroring, call the generic version
    if (!(padded.dim() == 4 && do_transform_0213) &&
        !(padded.dim() == 3 && !do_transform_0213)) {
      return at::native::nested_from_padded_generic(
          padded, sizes, do_transform_0213);
    }
    if (padded.dtype() != at::kFloat && padded.dtype() != kHalf) {
      TORCH_WARN_ONCE(
          "nested_from_padded XPU kernels only support fp32/fp16; falling "
          "back to slower generic kernel");
      return at::native::nested_from_padded_generic(
          padded, sizes, do_transform_0213);
    }
    Tensor target_offsets =
        at::native::NestedTensor_batch_offsets_from_size_tensor(sizes, 0);
    Tensor padded_sizes_tensor = at::tensor(padded.sizes());
    Tensor output = at::empty({padded_tensor_numel(sizes)}, padded.options());
    Tensor target_size_sizes = sizes.reshape(-1);

    Tensor metadata =
        at::cat({target_size_sizes, padded_sizes_tensor, target_offsets});
    metadata = metadata.to(at::Device(kXPU), kInt, true, true);

    auto output_size_ptr = metadata.data_ptr<int>();
    auto input_size_ptr = output_size_ptr + target_size_sizes.numel();
    auto offsets_ptr = input_size_ptr + padded_sizes_tensor.numel();

    Tensor padded_contiguous = padded.contiguous();
    if (padded.dtype() == at::kFloat) {
      if (do_transform_0213) {
        xpu::remove_padding_transform0213_kernel_float(
            padded_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 2,
            padded_contiguous.sizes()[0]);
      } else {
        xpu::remove_padding_kernel_float(
            padded_contiguous.data_ptr<float>(),
            output.data_ptr<float>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 1,
            padded_contiguous.sizes()[0]);
      }
    } else if (padded.dtype() == at::kHalf) {
      if (do_transform_0213) {
        xpu::remove_padding_transform0213_kernel_half(
            padded_contiguous.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 2,
            padded_contiguous.sizes()[0]);
      } else {
        xpu::remove_padding_kernel_half(
            padded_contiguous.data_ptr<c10::Half>(),
            output.data_ptr<c10::Half>(),
            offsets_ptr,
            input_size_ptr,
            output_size_ptr,
            padded_contiguous.dim() - 1,
            padded_contiguous.sizes()[0]);
      }
    } else {
      TORCH_CHECK(false, "Only support fp32/fp16 for padded input");
    }
    return at::detail::make_tensor<at::native::NestedTensorImpl>(
        std::move(output), sizes);
  } else {
    return at::native::nested_from_padded_generic(padded, sizes);
  }
}

static Tensor batch_offsets_from_efficient_size(const Tensor& ef_sizes) {
  const int64_t* nt_sizes_ptr = ef_sizes.const_data_ptr<int64_t>();
  int64_t ef_sizes_size_0 = ef_sizes.sizes()[0];
  Tensor offsets = at::empty({1 + ef_sizes_size_0}, at::kLong);
  int64_t* offsets_ptr = offsets.mutable_data_ptr<int64_t>();
  offsets_ptr[0] = 0;
  int64_t ef_sizes_size_1 = ef_sizes.sizes()[1];
  for (const auto i : c10::irange(ef_sizes_size_0)) {
    int64_t prod = 1;
    for (const auto j : c10::irange(ef_sizes_size_1)) {
      prod = prod * nt_sizes_ptr[i * ef_sizes_size_1 + j];
    }
    offsets_ptr[i + 1] = offsets_ptr[i] + prod;
  }
  return offsets;
}

Tensor NestedTensor_to_padded_tensor_xpu(
    const Tensor& t,
    double padding,
    OptionalIntArrayRef output_size) {
  TORCH_CHECK(
      t.numel() > 0,
      "to_padded_tensor: at least one constituent tensor should have non-zero numel")
  int64_t t_dim = t.dim();
  if (t_dim >= 2 && t_dim <= 4 &&
      (t.dtype() == at::kFloat || t.dtype() == at::kDouble ||
       t.dtype() == at::kHalf)) {
    auto* nt_input = get_nested_tensor_impl(t);
    TORCH_CHECK(
        nested_tensor_impl_is_contiguous(nt_input),
        "for now to_padded_tensor only supports contiguous nested tensor");
    const auto& nt_buffer = nt_input->get_buffer();

    if (t_dim == 3 && !output_size.has_value() && nt_input->opt_size(2) > 0) {
      const int64_t size_dim2 = *nt_input->opt_size(2);
      Tensor nt_sizes = nt_input->get_nested_sizes();
      Tensor sizes_dim1 = at::native::narrow_symint(nt_sizes, 1, 0, 1);
      Tensor sizes_dim2 = at::native::narrow_symint(nt_sizes, 1, 1, 1);
      Tensor result = at::detail::make_tensor<NestedTensorImpl>(
          nt_input->get_buffer(), sizes_dim1 * sizes_dim2[0]);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.dim() == 2);
      result = NestedTensor_to_padded_tensor_xpu(result, padding, output_size);
      return result.reshape({result.sizes()[0], -1, size_dim2});
    }

    Tensor nt_sizes = nt_input->get_nested_sizes();
    Tensor offsets = batch_offsets_from_efficient_size(nt_sizes);
    auto new_size = NestedTensor_get_max_size(*nt_input);
    new_size.insert(new_size.begin(), nt_sizes.sizes()[0]);

    // Pad output tensor to output_size if provided
    if (output_size.has_value()) {
      auto output_size_ = output_size.value();
      TORCH_CHECK(
          output_size_.size() == new_size.size(),
          "Length of output_size does not match NestedTensor dims. Broadcasting is not supported.");
      for (uint64_t i = 0; i < new_size.size(); i++) {
        TORCH_CHECK(
            output_size_[i] >= new_size[i],
            "Value in output_size is less than NestedTensor padded size. Truncation is not supported.");
        new_size[i] = output_size_[i];
      }
    }

    Tensor output = at::empty(IntArrayRef(new_size), nt_buffer.options());

    int64_t input_dim = nt_sizes.sizes()[1];
    int64_t batch_size = nt_sizes.sizes()[0];
    int64_t output_batch_size = new_size[0];
    // TODO: Remove need for cat here
    at::Tensor metadata = at::cat({offsets, nt_sizes.reshape(-1)});
    metadata = metadata.to(at::Device(kXPU), at::kInt);

    std::vector<Tensor> split =
        at::split_with_sizes(metadata, {offsets.numel(), nt_sizes.numel()}, 0);

    offsets = split[0];
    nt_sizes = split[1];

    xpu::add_padding_kernel(
        nt_buffer,
        output,
        padding,
        offsets,
        nt_sizes,
        input_dim,
        new_size,
        batch_size,
        output_batch_size);

    return output;
  }
  return NestedTensor_to_padded_tensor_generic(t, padding, output_size);
}

at::Tensor _jagged_to_padded_dense_forward_xpu(
    const Tensor& values,
    TensorList offsets,
    c10::IntArrayRef max_lengths,
    const double padding_value) {
  const size_t num_jagged_dim = offsets.size();

  TORCH_CHECK(
      max_lengths.size() == num_jagged_dim,
      "max_lengths.size(), ",
      max_lengths.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);
  c10::OptionalDeviceGuard device_guard;
  device_guard.reset_device(values.device());

  return at::native::xpu::jagged_to_padded_dense_forward_xpu_kernel(
      values, offsets, max_lengths, padding_value);
}

Tensor _padded_dense_to_jagged_forward_symint_xpu(
    const Tensor& padded,
    TensorList offsets_list,
    std::optional<c10::SymInt> total_L) {
  c10::OptionalDeviceGuard device_guard;
  device_guard.reset_device(padded.device());

  return at::native::xpu::dense_to_jagged_forward_kernel(
      padded, offsets_list, total_L);
}

// Materializes one SDPA operand, laid out as (batch, num_heads, {seq_len},
// head_dim), at the attention's output batch and num_heads. CUDA instead folds
// the broadcast into strides over the nested buffer
// (sdpa_nested_preprocessing_with_broadcast in
// aten/src/ATen/native/nested/cuda/NestedTensorTransformerUtils.cpp:210-365),
// because its varlen kernels consume a packed (Nnz, num_heads, head_dim) view.
// XPU has no varlen kernel, so the operands have to end up as plain
// equally-shaped contiguous nested tensors for the fallbacks below.
static Tensor broadcast_nested_sdpa_input(
    const Tensor& t,
    int64_t batch_size,
    int64_t num_heads) {
  // The operands are (batch, num_heads, seq_len, head_dim) views over
  // (batch, seq_len, num_heads, head_dim) nested storage whose metadata can
  // still carry the transpose. Transposing back is a metadata identity and
  // gives the physically packed (seq_len, num_heads, head_dim) constituents,
  // which are permuted and broadcast as plain dense tensors to rebuild a
  // self-consistent (batch, num_heads, seq_len, head_dim) nested tensor.
  auto constituents = t.transpose(1, 2).unbind();
  std::vector<Tensor> aligned;
  aligned.reserve(constituents.size());
  for (const auto& seq : constituents) {
    aligned.push_back(seq.transpose(0, 1).expand({num_heads, -1, -1}));
  }
  if (t.size(0) == 1 && batch_size != 1) {
    TORCH_INTERNAL_ASSERT(constituents.size() == 1);
    auto seq = aligned[0];
    aligned.assign(batch_size, std::move(seq));
  }
  return at::_nested_tensor_from_tensor_list(aligned);
}

// Sequence lengths of a broadcast nested operand, read from its
// (batch, num_heads, seq_len, head_dim) nested sizes. The nested sizes view is
// not contiguous, so it is materialized before the lengths are read through a
// raw pointer.
static std::vector<int64_t> nested_seq_lens(const Tensor& t) {
  const auto lens_cpu = get_nested_tensor_impl(t)
                            ->get_nested_sizes()
                            .select(1, 1)
                            .contiguous()
                            .cpu();
  const int64_t* lens = lens_cpu.const_data_ptr<int64_t>();
  return std::vector<int64_t>(lens, lens + lens_cpu.numel());
}

/**
 * Fall back implementation of nested efficient attention, mirroring the dense
 * _scaled_dot_product_efficient_attention_xpu: XPU has no fused nested
 * attention kernel, so this broadcasts the operands, materializes them to
 * padded dense tensors, and runs the dense math kernel once per constituent.
 * The math kernel has no nested-tensor support on XPU, so it is called on the
 * plain dense per-constituent tensors instead of the broadcast operands.
 */
std::tuple<Tensor, Tensor, Tensor, Tensor>
_scaled_dot_product_efficient_attention_nestedtensor_xpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    [[maybe_unused]] bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  TORCH_CHECK(
      !attn_bias.has_value(),
      "Nested tensor inputs are not supported with an attn_mask on XPU yet");
  const int64_t batch_size =
      std::max({query.size(0), key.size(0), value.size(0)});
  const int64_t num_heads =
      std::max({query.size(1), key.size(1), value.size(1)});

  auto broadcasted_query =
      broadcast_nested_sdpa_input(query, batch_size, num_heads);
  auto broadcasted_key =
      broadcast_nested_sdpa_input(key, batch_size, num_heads);
  auto broadcasted_value =
      broadcast_nested_sdpa_input(value, batch_size, num_heads);

  // The nested sizes view is not contiguous, so materialize it before reading
  // the lengths through a raw pointer.
  const auto query_lens = nested_seq_lens(broadcasted_query);
  const auto key_lens = nested_seq_lens(broadcasted_key);
  const auto value_lens = nested_seq_lens(broadcasted_value);

  auto padded_query = broadcasted_query.to_padded_tensor(0);
  auto padded_key = broadcasted_key.to_padded_tensor(0);
  auto padded_value = broadcasted_value.to_padded_tensor(0);

  std::vector<Tensor> attention_constituents;
  attention_constituents.reserve(batch_size);
  for (const auto i : c10::irange(batch_size)) {
    // Dense kernels, not the dispatcher: the broadcast operands are nested,
    // so dispatching on them would recurse into this function.
    auto attention = std::get<0>(at::native::_scaled_dot_product_attention_math(
        padded_query.select(0, i).narrow(1, 0, query_lens[i]).unsqueeze(0),
        padded_key.select(0, i).narrow(1, 0, key_lens[i]).unsqueeze(0),
        padded_value.select(0, i).narrow(1, 0, value_lens[i]).unsqueeze(0),
        attn_bias,
        dropout_p,
        is_causal,
        std::nullopt, /*dropout_mask*/
        scale,
        false /*enable_gqa, num_heads already equalized*/));
    attention_constituents.push_back(attention.squeeze(0));
  }

  auto attention = at::_nested_tensor_from_tensor_list(attention_constituents);

  // The math kernel produces no logsumexp, and the seed/offset pair only exists
  // so backward can replay dropout. Neither is reachable: sdp_params rejects
  // nested inputs that require grad, so there is no backward for this path.
  return std::make_tuple(
      std::move(attention),
      at::empty({0}, query.options()),
      at::scalar_tensor(int64_t(0), query.options().dtype(at::kLong)),
      at::scalar_tensor(int64_t(0), query.options().dtype(at::kLong)));
}

/**
 * Fall back implementation of nested flash attention, mirroring the dense
 * _scaled_dot_product_flash_attention_xpu: XPU has no varlen flash kernel to
 * consume the packed buffers CUDA's nested preprocessing produces
 * (sdpa_nested_preprocessing), so this broadcasts the operands, materializes
 * them to padded dense tensors in (batch, num_heads, seq_len, head_dim)
 * order, and runs the dense kernel once per constituent. The output nested
 * tensor is built from the per-batch results, so it stays in the same
 * (batch, num_heads, seq_len, head_dim) order the dense kernel returns.
 */
std::tuple<
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    c10::SymInt,
    c10::SymInt,
    Tensor,
    Tensor,
    Tensor>
_scaled_dot_product_flash_attention_nestedtensor_xpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    [[maybe_unused]] bool return_debug_mask,
    std::optional<double> scale) {
  const int64_t batch_size =
      std::max({query.size(0), key.size(0), value.size(0)});
  const int64_t num_heads =
      std::max({query.size(1), key.size(1), value.size(1)});

  auto broadcasted_query =
      broadcast_nested_sdpa_input(query, batch_size, num_heads);
  auto broadcasted_key =
      broadcast_nested_sdpa_input(key, batch_size, num_heads);
  auto broadcasted_value =
      broadcast_nested_sdpa_input(value, batch_size, num_heads);

  // The nested sizes view is not contiguous, so materialize it before reading
  // the lengths through a raw pointer.
  const auto query_lens = nested_seq_lens(broadcasted_query);
  const auto key_lens = nested_seq_lens(broadcasted_key);
  const auto value_lens = nested_seq_lens(broadcasted_value);

  auto padded_query = broadcasted_query.to_padded_tensor(0);
  auto padded_key = broadcasted_key.to_padded_tensor(0);
  auto padded_value = broadcasted_value.to_padded_tensor(0);

  std::vector<Tensor> attention_constituents;
  attention_constituents.reserve(batch_size);
  for (const auto i : c10::irange(batch_size)) {
    // Dense kernels, not the dispatcher: the broadcast operands are nested,
    // so dispatching on them would recurse into this function.
    auto attention =
        std::get<0>(at::native::_scaled_dot_product_flash_attention_xpu(
            padded_query.select(0, i).narrow(1, 0, query_lens[i]).unsqueeze(0),
            padded_key.select(0, i).narrow(1, 0, key_lens[i]).unsqueeze(0),
            padded_value.select(0, i).narrow(1, 0, value_lens[i]).unsqueeze(0),
            dropout_p,
            is_causal,
            false, /*return_debug_mask, per-constituent masks do not merge*/
            scale));
    attention_constituents.push_back(attention.squeeze(0));
  }

  auto attention = at::_nested_tensor_from_tensor_list(attention_constituents);

  // The flash kernel produces no logsumexp here, and the seed/offset pair
  // only exists so backward can replay dropout. Neither is reachable: the
  // sdp gate rejects nested inputs that require grad or dropout, so there is
  // no backward for this path.
  return std::make_tuple(
      std::move(attention),
      at::empty({0}, query.options()),
      at::empty({0}, query.options().dtype(at::kInt)),
      at::empty({0}, query.options().dtype(at::kInt)),
      int64_t(0),
      int64_t(0),
      at::scalar_tensor(int64_t(0), query.options().dtype(at::kLong)),
      at::scalar_tensor(int64_t(0), query.options().dtype(at::kLong)),
      at::empty({0}, query.options()));
}

} // namespace at::native
