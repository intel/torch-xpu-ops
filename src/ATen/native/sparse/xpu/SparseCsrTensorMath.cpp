/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add.h>
#endif

namespace at::native {

using namespace at::sparse;
using namespace at::sparse_csr;

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

Tensor& add_out_sparse_compressed_xpu(
    const Tensor& self,
    const SparseCsrTensor& other,
    const Scalar& alpha,
    SparseCsrTensor& out) {
  if (self.layout() == kStrided) {
    at::add_out(out, self, other.to_dense(), alpha);
    return out;
  } else if (other.layout() == kStrided) {
    at::add_out(out, other, self.to_dense(), alpha);
    return out;
  } else {
    TORCH_CHECK(
        self.sizes().equals(other.sizes()),
        "torch.add: Expected input tensors to have the same shape, but got tensor `self` with shape ",
        self.sizes(),
        " and tensor `other` with shape ",
        other.sizes());
    TORCH_CHECK(
        self.is_xpu(),
        "add: expected 'self' to be XPU tensor, but got tensor on device: ",
        self.device());
    TORCH_CHECK(
        other.is_xpu(),
        "add: expected 'other' to be XPU tensor, but got tensor on device: ",
        other.device());
    TORCH_CHECK(
        out.is_xpu(),
        "add: expected 'out' to be XPU tensor, but got tensor on device: ",
        out.device());

    if (only_sparse_compressed_add_trivial_cases(self, other, alpha, out)) {
      return out;
    }

    Tensor out_dense = at::add(self.to_dense(), other.to_dense(), alpha);
    out = out_dense.to_sparse_csr();
  }
  return out;
}

} // namespace at::native
