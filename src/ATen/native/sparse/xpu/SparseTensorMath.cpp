/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/sparse/xpu/sycl/SparseTensorMathKernels.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm.h>
#include <ATen/ops/matmul.h>
#endif

#include <ATen/ExpandUtils.h>

namespace at::native {

using namespace at::sparse;

SparseTensor& add_out_sparse_xpu(
    const SparseTensor& t,
    const SparseTensor& src,
    const Scalar& value,
    SparseTensor& r_) {
  return xpu::add_sparse_kernel(t, src, value, r_);
}

SparseTensor& mul_out_sparse_xpu(
    const Tensor& t_,
    const Tensor& src_,
    SparseTensor& r_) {
  return xpu::mul_sparse_kernel(t_, src_, r_);
}

Tensor _sparse_sum_backward_xpu(
    const Tensor& grad_,
    const SparseTensor& input_,
    IntArrayRef dims_to_sum) {
  return xpu::_sparse_sum_backward_kernel(grad_, input_, dims_to_sum);
}

Tensor& s_addmm_out_sparse_dense_xpu(Tensor& r_, const Tensor& t, const SparseTensor& sparse_, const Tensor& dense, const Scalar& beta, const Scalar& alpha) {
  TORCH_CHECK(t.is_xpu(), "Expected all tensors to be on the same device. addmm: expected 'self' to be XPU, but got CPU");
  TORCH_CHECK(r_.is_xpu(), "Expected all tensors to be on the same device. addmm: expected 'out' to be XPU, but got CPU");
  TORCH_CHECK(sparse_.is_xpu(), "Expected all tensors to be on the same device. addmm: expected 'mat1' to be XPU, but got CPU");
  TORCH_CHECK(dense.is_xpu(), "Expected all tensors to be on the same device. addmm: expected 'mat2' to be XPU, but got CPU");

  // TORCH_CHECK(xpu::check_device({sparse_, r_, t, dense}));

  TORCH_CHECK(dense.dim() == 2, "addmm: 2D tensor expected, got ", dense.dim(), "D tensor");
  TORCH_CHECK(sparse_.sparse_dim() == 2, "addmm: expected first two dims to be sparse (indices has size 2 at first dim), but got ", sparse_.sparse_dim(), " sparse dims");
  // no need to check dense_dim because dense_dim + sparse_dim = dim

  Tensor mat1_dense = sparse_._to_dense(std::nullopt, std::nullopt);
  at::addmm_out(r_, t, mat1_dense, dense, beta, alpha);

  return r_;
}

Tensor s_addmm_sparse_dense_xpu(
    const Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha
) {
  Tensor r = at::empty({0}, t.options());
  s_addmm_out_sparse_dense_xpu(r, t, sparse, dense, beta, alpha);
  return r;
}


Tensor& addmm_out_sparse_dense_xpu(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result
) {
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_out_sparse_dense_xpu(result, *b_self, mat1, mat2, beta, alpha);
}

Tensor addmm_sparse_dense_xpu(
    const Tensor& self,
    const SparseTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha
) {
  c10::MaybeOwned<Tensor> b_self = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm_out");
  return s_addmm_sparse_dense_xpu(*b_self, mat1, mat2, beta, alpha);
}

Tensor& s_addmm_sparse_dense_xpu_(
    Tensor& t,
    const SparseTensor& sparse,
    const Tensor& dense,
    const Scalar& beta,
    const Scalar& alpha
) {
  return s_addmm_out_sparse_dense_xpu(t, t, sparse, dense, beta, alpha);
}

Tensor sparse_sparse_matmul_xpu(const Tensor& mat1_, const Tensor& mat2_) {
  TORCH_INTERNAL_ASSERT(mat1_.is_sparse());
  TORCH_INTERNAL_ASSERT(mat2_.is_sparse());
  TORCH_CHECK(mat1_.dim() == 2);
  TORCH_CHECK(mat2_.dim() == 2);
  TORCH_CHECK(mat1_.dense_dim() == 0, "sparse_mm: scalar values expected, mat1 got ", mat1_.dense_dim(), "D values");
  TORCH_CHECK(mat2_.dense_dim() == 0, "sparse_mm: scalar values expected, mat2 got ", mat2_.dense_dim(), "D values");

  TORCH_CHECK(
      mat1_.size(1) == mat2_.size(0), "mat1 and mat2 shapes cannot be multiplied (",
      mat1_.size(0), "x", mat1_.size(1), " and ", mat2_.size(0), "x", mat2_.size(1), ")");

  TORCH_CHECK(mat1_.scalar_type() == mat2_.scalar_type(),
           "mat1 dtype ", mat1_.scalar_type(), " does not match mat2 dtype ", mat2_.scalar_type());
  
  // convert to dense
  Tensor mat1_dense = mat1_._to_dense(std::nullopt, std::nullopt);
  Tensor mat2_dense = mat2_._to_dense(std::nullopt, std::nullopt);

  Tensor output_dense = at::matmul(mat1_dense, mat2_dense);
  // convert back to sparse
  Tensor output_sparse = output_dense._to_sparse(mat1.layout());

  return output_sparse;

  // auto output = at::native::empty_like(mat1_);
  // output.sparse_resize_and_clear_({mat1_.size(0), mat2_.size(1)}, mat1_.sparse_dim(), 0);
}

} // namespace at::native
