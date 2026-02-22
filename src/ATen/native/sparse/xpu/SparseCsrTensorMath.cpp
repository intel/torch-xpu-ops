/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ExpandUtils.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmv.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/scalar_tensor_native.h>
#endif

namespace at::native {

using namespace at::sparse;

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_xpu)
(const Tensor& input,
 const int64_t size,
 const bool out_int32,
 const Tensor& result) {
  xpu::convert_indices_from_coo_to_csr_structured_kernel(
      input, size, out_int32, result);
};

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_xpu)
(const Tensor& crow_indices,
 const Tensor& col_indices,
 const bool out_int32,
 const bool transpose,
 const Tensor& result) {
  xpu::convert_indices_from_csr_to_coo_structured_kernel(
      crow_indices, col_indices, out_int32, transpose, result);
};

Tensor _sparse_csr_sum_xpu(
    const Tensor& input,
    IntArrayRef dims_to_sum,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  return xpu::_sparse_csr_sum_xpu_kernel(input, dims_to_sum, keepdim, dtype);
}

Tensor _sparse_csr_prod_xpu(
    const Tensor& input,
    IntArrayRef dims_to_reduce,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  return xpu::_sparse_csr_prod_xpu_kernel(
      input, dims_to_reduce, keepdim, dtype);
}

Tensor& addmv_out_sparse_compressed_xpu(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  if (mat.layout() == kSparseCsc) {
    return addmv_out_sparse_compressed_xpu(
        self, mat.to_sparse_csr(), vec, beta, alpha, result);
  }
  TORCH_CHECK(
      mat.layout() != kSparseBsc,
      "addmv_out_sparse_compressed_xpu currently does not support layout SparseBsc for input mat.");

  TORCH_CHECK(mat.dim() == 2, "addmv: Expected mat to be 2-D");
  TORCH_CHECK(vec.dim() == 1, "addmv: Expected vec to be 1-D");

  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta.toComplexDouble();

  if (&result != &self) {
    at::native::resize_output(result, self_->sizes());
    if (betaval != 0.0) {
      at::native::copy_(result, *self_);
    }
  }

  if (mat._nnz() == 0) {
    // shortcut for an empty matrix
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (betaval == 0.0) {
      return result.zero_();
    } else {
      return at::mul_out(
          result,
          self,
          at::native::scalar_tensor(
              beta,
              self.scalar_type(),
              std::nullopt /* layout */,
              at::kXPU,
              std::nullopt /* pin_memory */));
    }
  }

  at::addmv_out(result, self, mat.to_dense(), vec, beta, alpha);

  return result;
}

} // namespace at::native
