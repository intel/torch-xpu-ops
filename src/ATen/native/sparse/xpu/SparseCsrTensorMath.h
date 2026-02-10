/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Scalar.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/cpu/SpmmReduceKernel.h>

namespace at::native::sparse::impl {

// Returns true if all entries of self are zero
// TODO: This has potential to be a generic helper
inline bool _is_sparse_and_zero(const Tensor& self) {
  if (self.layout() == kSparse || self.layout() == kSparseCsr ||
      self.layout() == kSparseCsc || self.layout() == kSparseBsr ||
      self.layout() == kSparseBsc) {
    if (self._nnz() == 0) {
      return true;
    }
  }
  return false;
}

inline void _check_is_cpu(const Tensor& self, std::string_view name) {
  TORCH_CHECK(
      self.is_cpu(),
      "Expected all tensors to be on the same device. addmm expected '",
      name,
      "' to be CPU tensor, but got ",
      self.device(),
      " tensor");
}

inline void _check_is_xpu(const Tensor& self, std::string_view name) {
  TORCH_CHECK(
      self.is_xpu(),
      "Expected all tensors to be on the same device. addmm expected '",
      name,
      "' to be XPU tensor, but got ",
      self.device(),
      " tensor");
}

inline void _check_dim(
    const Tensor& self,
    int64_t target_dim,
    std::string_view name) {
  if (target_dim == 2) {
    TORCH_CHECK(
        self.dim() == target_dim,
        name,
        " must be a matrix, ",
        "got ",
        self.dim(),
        "-D tensor");
  } else {
    TORCH_CHECK(
        self.dim() == target_dim,
        "Expected ",
        name,
        " to be of dimension ",
        target_dim,
        " but got ",
        self.dim(),
        " instead.");
  }
}

} // namespace at::native::sparse::impl
