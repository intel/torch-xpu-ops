/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/SparseBlas.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/add.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/index.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/sgn.h>
#include <ATen/ops/sparse_sampled_addmm_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#endif

// #include <c10/util/MaybeOwned.h>

namespace at::native {

/*
  Computes `result` <- α*(A @ B) * spy(C) + β*C, where spy(C) is the sparsity
  pattern matrix of C.

  Args:
  * `mat1` - [in] dense Tensor A of size m × k.
  * `mat2` - [in] dense Tensor B of size k × n.
  * `self` - [in] sparse Tensor C of size m × n.
  * `result` - [out] sparse Tensor of size m × n.
*/
Tensor& sparse_sampled_addmm_out_sparse_csr_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  at::native::sparse::sparse_sampled_addmm_check_inputs(
      self, mat1, mat2, beta, alpha, result);

  // resize result if have batch
  if (&result != &self) {
    // We allow self to be a single matrix when mat1 and mat2 are batched
    auto result_sizes = DimVector(mat1.sizes().slice(0, mat1.dim() - 2));
    result_sizes.push_back(self.size(-2));
    result_sizes.push_back(self.size(-1));
    at::sparse_csr::get_sparse_csr_impl(result)->resize_(
        self._nnz(), result_sizes);
    result.copy_(self);
  }

  if (mat1.numel() == 0 || mat2.numel() == 0 || result._nnz() == 0) {
    result.mul_(beta);
    return result;
  }

  auto sizes = result.sizes();
  auto M = self.size(-2);
  auto N = self.size(-1);
  auto total_batch = result.numel() / (M * N);
  auto crow = result.crow_indices(); // [B1, B2, ..., M+1]
  auto col = result.col_indices(); // [B1, B2, ..., nnz_per_batch]
  auto values = result.values(); // [B1, B2, ..., nnz_per_batch]

  // elment counts of all rows [B1, B2, ..., M]
  auto row_counts =
      (crow.narrow(-1, 1, M) - crow.narrow(-1, 0, M)).reshape((-1));

  // row_base of [0, 1, ..., M-1] to flat to all batches
  auto row_base = at::arange(M, crow.options()).repeat((total_batch));
  auto row_indices = at::repeat_interleave(row_base, row_counts);

  auto nnz_per_batch = (crow.select(-1, M) - crow.select(-1, 0)).reshape({-1});
  auto batch_indices_flat = at::repeat_interleave(
      at::arange(total_batch, crow.options()), nnz_per_batch);

  // restore to Multi-dim Batch Indices
  torch::List<std::optional<Tensor>> batch_coords;
  auto temp_batch_idx = batch_indices_flat;
  auto batch_sizes = sizes.slice(0, result.dim() - 2); // [B1, B2, ...]
  for (int i = batch_sizes.size() - 1; i >= 0; --i) {
    batch_coords.insert(
        batch_coords.begin(),
        std::optional<Tensor>(temp_batch_idx % batch_sizes[i]));
    temp_batch_idx = temp_batch_idx.divide(batch_sizes[i], "trunc");
  }

  // get all row indices [B1_idx, B2_idx, Row_idx]
  torch::List<std::optional<Tensor>> a_indices;
  a_indices.append(batch_coords);
  a_indices.push_back(std::optional<Tensor>(row_indices));
  auto a_sub = mat1.index(a_indices); // (Total_NNZ, K)
  // get all col indices [B1_idx, B2_idx, Col_idx]
  torch::List<std::optional<Tensor>> b_indices;
  b_indices.append(batch_coords);
  b_indices.push_back(std::optional<Tensor>(col.reshape({-1})));
  auto b_sub = mat2.transpose(-1, -2).index(b_indices); // (Total_NNZ, K)

  // dot product (Total_NNZ)
  auto dot_products = (a_sub * b_sub).sum(-1);

  auto new_values = (dot_products * alpha) + (values.reshape({-1}) * beta);

  auto result_temp = at::native::_sparse_csr_tensor_unsafe(
      crow,
      col,
      new_values.view_as(values),
      sizes,
      new_values.scalar_type(),
      self.layout(),
      new_values.device());

  result.copy_(result_temp);
  return result;
}

Tensor sparse_sampled_addmm_sparse_csr_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  auto result = at::empty({0, 0}, self.options());
  at::native::sparse_sampled_addmm_out_sparse_csr_xpu(
      self, mat1, mat2, beta, alpha, result);
  return result;
}

} // namespace at::native
