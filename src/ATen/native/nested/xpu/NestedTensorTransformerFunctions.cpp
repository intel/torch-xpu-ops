/*
 * Copyright 2020-2025 Intel Corporation
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

namespace at::native {

namespace {

int64_t padded_tensor_numel(const Tensor& sizes) {
  const auto sizes_num_rows = sizes.sizes()[0];
  const auto sizes_row_length = sizes.sizes()[1];
  const auto* sizes_data = sizes.data_ptr<int64_t>();
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
  int64_t* nt_sizes_ptr = ef_sizes.data_ptr<int64_t>();
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

    if (t_dim == 3 && nt_input->opt_size(2) && (*nt_input->opt_size(2) > 0) &&
        !(output_size.has_value())) {
      Tensor nt_sizes = nt_input->get_nested_sizes();
      Tensor sizes_dim1 = at::native::narrow_symint(nt_sizes, 1, 0, 1);
      Tensor sizes_dim2 = at::native::narrow_symint(nt_sizes, 1, 1, 1);
      Tensor result = at::detail::make_tensor<NestedTensorImpl>(
          nt_input->get_buffer(), sizes_dim1 * sizes_dim2[0]);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(result.dim() == 2);
      result = NestedTensor_to_padded_tensor_xpu(result, padding, output_size);
      return result.reshape({result.sizes()[0], -1, *nt_input->opt_size(2)});
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

at::Tensor _fbgemm_jagged_to_padded_dense_forward(
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

  return at::native::xpu::_fbgemm_jagged_to_padded_dense_forward_kernel(
      values, offsets, max_lengths, padding_value);
}

Tensor _padded_dense_to_jagged_forward_xpu(
    const Tensor& padded,
    TensorList offsets_list,
    std::optional<int64_t> total_L) {
  // TODO: Make this kernel more efficient using TensorIterator or something.
  TORCH_INTERNAL_ASSERT(
      offsets_list.size() == 1,
      "_padded_dense_to_jagged_forward(): only a single jagged dim is supported for now");

  // allocate appropriately-sized values tensor
  const auto& offsets = offsets_list[0];
  TORCH_CHECK(
      offsets.dim() == 1,
      "_padded_dense_to_jagged_forward(): expected 1D offsets, but got offsets.dim() == ",
      offsets.dim());

  auto final_offset = offsets[-1].item<int64_t>();
  int64_t total_L_val = total_L.has_value() ? (*total_L) : final_offset;
  if (total_L.has_value()) {
    // error if the offsets try to index past the end of the packed dimension
    TORCH_CHECK(
        final_offset == total_L_val,
        "_padded_dense_to_jagged_forward(): final offset should match total_L value");
  }

  TORCH_CHECK(
      padded.dim() >= 2,
      "_padded_dense_to_jagged_forward(): expected padded dim >= 2, but padded.dim() == ",
      padded.dim());

  std::vector<int64_t> values_shape;
  values_shape.reserve(padded.dim() - 1);
  values_shape.push_back(total_L_val);
  auto padded_shape = padded.sizes();
  values_shape.insert(values_shape.end(), padded_shape.begin() + 2, padded_shape.end());
  Tensor values = padded.new_empty(values_shape);

  // copy data to values tensor
  auto batch_size = offsets.size(0) - 1;
  for (auto i : c10::irange(batch_size)) {
    auto start_offset = offsets[i].item<int64_t>();
    auto end_offset = offsets[i + 1].item<int64_t>();
    auto length = end_offset - start_offset;

    TORCH_CHECK(
        length <= padded_shape[1],
        "_padded_dense_to_jagged_forward(): found batch item of length ", length,
        " when max length specified by padded input is ", padded_shape[1]);

    auto dst = values.slice(0, start_offset, end_offset);
    auto source = padded.select(0, i).slice(0, 0, length);
    dst.copy_(source);
  }

  return values;
}

} // namespace at::native
