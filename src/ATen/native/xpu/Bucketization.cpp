/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/native/BucketizationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/BucketizationKernels.h>

namespace at {
namespace native {

Tensor& searchsorted_out_xpu(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt,
    Tensor& result) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> sorter_maybe_owned =
      at::borrow_from_optional_tensor(sorter_opt);
  const Tensor& sorter = *sorter_maybe_owned;
  searchsorted_pre_check(
      sorted_sequence, self, result, out_int32, right, side_opt, sorter);
  resize_output(result, self.sizes());

  if (self.numel() == 0) {
    return result;
  }

  // we have two inputs to set right, pre_check checks that they aren't set to
  // opposites
  bool is_right = (side_opt && *side_opt == "right") || right;
  xpu::searchsorted_kernel(
      result, self, sorted_sequence, out_int32, is_right, sorter);
  return result;
}

Tensor& searchsorted_out_xpu(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter_opt,
    Tensor& result) {
  const Tensor& scalar_tensor =
      searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_out_xpu(
      sorted_sequence,
      scalar_tensor,
      out_int32,
      right,
      side_opt,
      sorter_opt,
      result);
}

Tensor searchsorted_xpu(
    const Tensor& sorted_sequence,
    const Tensor& self,
    bool out_int32,
    bool right,
    const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  searchsorted_out_xpu(
      sorted_sequence, self, out_int32, right, side_opt, sorter, result);
  return result;
}

Tensor searchsorted_xpu(
    const Tensor& sorted_sequence,
    const Scalar& self,
    bool out_int32,
    bool right,
    const std::optional<std::string_view> side_opt,
    const std::optional<Tensor>& sorter) {
  const Tensor& scalar_tensor =
      searchsorted_scalar_tensor(self, sorted_sequence.device());
  return searchsorted_xpu(
      sorted_sequence, scalar_tensor, out_int32, right, side_opt, sorter);
}

Tensor& bucketize_out_xpu(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right,
    Tensor& result) {
  TORCH_CHECK(
      boundaries.dim() == 1,
      "boundaries tensor must be 1 dimension, but got dim(",
      boundaries.dim(),
      ")");
  searchsorted_out_xpu(
      boundaries, self, out_int32, right, std::nullopt, std::nullopt, result);
  return result;
}

Tensor bucketize_xpu(
    const Tensor& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  ScalarType scalar_type = out_int32 ? ScalarType::Int : ScalarType::Long;
  c10::TensorOptions options =
      TensorOptions().device(self.options().device()).dtype(scalar_type);
  Tensor result = at::empty({0}, options, MemoryFormat::Contiguous);
  bucketize_out_xpu(self, boundaries, out_int32, right, result);
  return result;
}

Tensor bucketize_xpu(
    const Scalar& self,
    const Tensor& boundaries,
    bool out_int32,
    bool right) {
  return bucketize_xpu(
      searchsorted_scalar_tensor(self, boundaries.device()),
      boundaries,
      out_int32,
      right);
}
} // namespace native
} // namespace at
