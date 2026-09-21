/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/sparse/xpu/mkl/SparseTensorMath.h>

#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/zeros.h>

#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <oneapi/mkl/spblas.hpp>

namespace at::native::xpu {

Tensor _sspaddmm_mkl_out(
    const Tensor& row_indices,
    const Tensor& col_indices,
    const Tensor& values1,
    const Tensor& mat2,
    const Tensor& self,
    const Scalar& beta,
    const Scalar& alpha,
    int64_t dim_i,
    int64_t dim_j,
    int64_t dim_k,
    int64_t nnz1) {
  Tensor crow_indices =
      at::_convert_indices_from_coo_to_csr(row_indices, dim_i, false);
  Tensor mat2_contiguous = mat2.contiguous();
  Tensor dense_result = (beta.to<double>() != 0.0 && self._nnz() > 0)
      ? self.to_dense()
      : at::zeros({dim_i, dim_k}, mat2.options());

  auto queue = at::xpu::getCurrentSYCLQueue();
  oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&handle);

  auto run_mkl = [&](auto scalar) {
    using scalar_t = decltype(scalar);
    (void)scalar;
    auto set_data_event = oneapi::mkl::sparse::set_csr_data(
        queue,
        handle,
        dim_i,
        dim_j,
        nnz1,
        oneapi::mkl::index_base::zero,
        crow_indices.data_ptr<int64_t>(),
        col_indices.data_ptr<int64_t>(),
        values1.data_ptr<scalar_t>());
    auto optimize_event = oneapi::mkl::sparse::optimize_gemm(
        queue,
        oneapi::mkl::layout::row_major,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        handle,
        dim_k,
        {set_data_event});
    auto gemm_event = oneapi::mkl::sparse::gemm(
        queue,
        oneapi::mkl::layout::row_major,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        alpha.to<scalar_t>(),
        handle,
        mat2_contiguous.data_ptr<scalar_t>(),
        dim_k,
        dim_k,
        beta.to<scalar_t>(),
        dense_result.data_ptr<scalar_t>(),
        dim_k,
        {optimize_event});
    gemm_event.wait();
  };
  if (values1.scalar_type() == at::kFloat) {
    run_mkl(float{});
  } else {
    run_mkl(double{});
  }
  oneapi::mkl::sparse::release_matrix_handle(queue, &handle);

  return dense_result;
}

} // namespace at::native::xpu
