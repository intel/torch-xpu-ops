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
#include <ATen/Dispatch.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorAddKernels.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

using namespace at::sparse_csr;

// Pass 1 kernel: compute per-row output nnz by merging sorted column arrays.
template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void compute_row_nnz_kernel(
    const index_t* a_crow,
    const index_t* a_col,
    const index_t* b_crow,
    const index_t* b_col,
    index_t* row_nnz,
    int64_t nrows) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int64_t row = item.get_global_linear_id();
  if (row >= nrows) {
    return;
  }

  index_t a_start = a_crow[row];
  index_t a_end = a_crow[row + 1];
  index_t b_start = b_crow[row];
  index_t b_end = b_crow[row + 1];

  index_t i = a_start, j = b_start;
  index_t count = 0;
  while (i < a_end && j < b_end) {
    if (a_col[i] < b_col[j]) {
      ++i;
    } else if (a_col[i] > b_col[j]) {
      ++j;
    } else {
      ++i;
      ++j;
    }
    ++count;
  }
  count += (a_end - i) + (b_end - j);
  row_nnz[row] = count;
}

// Pass 2 kernel: merge column indices and combine values.
template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void merge_rows_kernel(
    const index_t* a_crow,
    const index_t* a_col,
    const scalar_t* a_val,
    const index_t* b_crow,
    const index_t* b_col,
    const scalar_t* b_val,
    const index_t* out_crow,
    index_t* out_col,
    scalar_t* out_val,
    scalar_t alpha,
    int64_t nrows) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int64_t row = item.get_global_linear_id();
  if (row >= nrows) {
    return;
  }

  index_t a_start = a_crow[row];
  index_t a_end = a_crow[row + 1];
  index_t b_start = b_crow[row];
  index_t b_end = b_crow[row + 1];
  index_t out_pos = out_crow[row];

  index_t i = a_start, j = b_start;
  while (i < a_end && j < b_end) {
    if (a_col[i] < b_col[j]) {
      out_col[out_pos] = a_col[i];
      out_val[out_pos] = a_val[i];
      ++i;
    } else if (a_col[i] > b_col[j]) {
      out_col[out_pos] = b_col[j];
      out_val[out_pos] = alpha * b_val[j];
      ++j;
    } else {
      out_col[out_pos] = a_col[i];
      out_val[out_pos] = a_val[i] + alpha * b_val[j];
      ++i;
      ++j;
    }
    ++out_pos;
  }
  while (i < a_end) {
    out_col[out_pos] = a_col[i];
    out_val[out_pos] = a_val[i];
    ++i;
    ++out_pos;
  }
  while (j < b_end) {
    out_col[out_pos] = b_col[j];
    out_val[out_pos] = alpha * b_val[j];
    ++j;
    ++out_pos;
  }
}

// Compute out = A + alpha * B for 2-D CSR tensors using a two-pass
// sorted-merge strategy (one work-item per row).
// Assumes column indices within each row are sorted (CSR format invariant).
void add_out_sparse_csr_kernel(
    const Tensor& A,
    const Tensor& B,
    const Scalar& alpha,
    Tensor& out) {
  TORCH_INTERNAL_ASSERT(
      A.dim() == 2 && B.dim() == 2,
      "add_out_sparse_csr_kernel: only 2-D CSR tensors are supported, "
      "got A.dim()=",
      A.dim(),
      " B.dim()=",
      B.dim());
  auto output_indices_dtype = promoteTypes(
      A.crow_indices().scalar_type(), B.crow_indices().scalar_type());
  auto output_values_dtype = promoteTypes(A.scalar_type(), B.scalar_type());
  int64_t nrows = A.size(0);

  // Handle empty (0 x N) tensors: sycl::range<1>(0) is invalid.
  if (nrows == 0) {
    auto idx_opts =
        at::TensorOptions().device(A.device()).dtype(output_indices_dtype);
    auto val_opts =
        at::TensorOptions().device(A.device()).dtype(output_values_dtype);
    Tensor out_crow = at::zeros({1}, idx_opts);
    Tensor out_col = at::empty({0}, idx_opts);
    Tensor out_val = at::empty({0}, val_opts);
    TORCH_INTERNAL_ASSERT(out.is_sparse_csr());
    static_cast<SparseCsrTensorImpl*>(out.unsafeGetTensorImpl())
        ->set_member_tensors(out_crow, out_col, out_val, A.sizes());
    return;
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      output_values_dtype, "add_out_sparse_csr_xpu", [&] {
        scalar_t alpha_val = alpha.to<scalar_t>();

        AT_DISPATCH_INDEX_TYPES(
            output_indices_dtype, "add_out_sparse_csr_xpu_index", [&] {
              auto a_crow = A.crow_indices().to(output_indices_dtype);
              auto a_col = A.col_indices().to(output_indices_dtype);
              auto b_crow = B.crow_indices().to(output_indices_dtype);
              auto b_col = B.col_indices().to(output_indices_dtype);
              auto a_val = A.values().to(output_values_dtype);
              auto b_val = B.values().to(output_values_dtype);

              const index_t* a_crow_ptr = a_crow.const_data_ptr<index_t>();
              const index_t* a_col_ptr = a_col.const_data_ptr<index_t>();
              const index_t* b_crow_ptr = b_crow.const_data_ptr<index_t>();
              const index_t* b_col_ptr = b_col.const_data_ptr<index_t>();
              const scalar_t* a_val_ptr = a_val.const_data_ptr<scalar_t>();
              const scalar_t* b_val_ptr = b_val.const_data_ptr<scalar_t>();

              // Pass 1: compute per-row output nnz.
              auto dense_idx_options = at::TensorOptions()
                                           .device(A.device())
                                           .dtype(output_indices_dtype);
              auto dense_val_options = at::TensorOptions()
                                           .device(A.device())
                                           .dtype(output_values_dtype);
              Tensor row_nnz = at::empty({nrows}, dense_idx_options);
              index_t* row_nnz_ptr = row_nnz.data_ptr<index_t>();

              constexpr auto pass1_kernel = compute_row_nnz_kernel<index_t>;
              int64_t pass1_group_size = syclMaxWorkGroupSize<pass1_kernel>();
              int64_t pass1_group_num =
                  (nrows + pass1_group_size - 1) / pass1_group_size;
              sycl_kernel_submit<pass1_kernel>(
                  sycl::range<1>(pass1_group_num * pass1_group_size),
                  sycl::range<1>(pass1_group_size),
                  getCurrentSYCLQueue(),
                  0,
                  a_crow_ptr,
                  a_col_ptr,
                  b_crow_ptr,
                  b_col_ptr,
                  row_nnz_ptr,
                  nrows);

              // Build crow_indices (size nrows + 1) via inclusive scan of
              // row_nnz into out_crow[1..nrows], then setting out_crow[0] = 0.
              // This produces the same result as an exclusive prefix sum.
              Tensor out_crow = at::empty({nrows + 1}, dense_idx_options);
              index_t* out_crow_ptr = out_crow.data_ptr<index_t>();

              pstl::inclusive_scan<index_t>(
                  row_nnz_ptr,
                  row_nnz_ptr + nrows,
                  out_crow_ptr + 1,
                  index_t(0));
              // Set out_crow[0] = 0 on device (no wait needed; the
              // in-order queue guarantees this completes before the
              // memcpy below).
              getCurrentSYCLQueue().memset(out_crow_ptr, 0, sizeof(index_t));

              // Read total nnz from device.
              index_t total_nnz = 0;
              getCurrentSYCLQueue()
                  .memcpy(&total_nnz, out_crow_ptr + nrows, sizeof(index_t))
                  .wait();

              // Allocate output col_indices and values.
              Tensor out_col = at::empty({total_nnz}, dense_idx_options);
              Tensor out_val = at::empty({total_nnz}, dense_val_options);

              if (total_nnz > 0) {
                index_t* out_col_ptr = out_col.data_ptr<index_t>();
                scalar_t* out_val_ptr = out_val.data_ptr<scalar_t>();

                // Pass 2: merge rows and fill col_indices + values.
                constexpr auto pass2_kernel =
                    merge_rows_kernel<scalar_t, index_t>;
                int64_t pass2_group_size = syclMaxWorkGroupSize<pass2_kernel>();
                int64_t pass2_group_num =
                    (nrows + pass2_group_size - 1) / pass2_group_size;
                sycl_kernel_submit<pass2_kernel>(
                    sycl::range<1>(pass2_group_num * pass2_group_size),
                    sycl::range<1>(pass2_group_size),
                    getCurrentSYCLQueue(),
                    0,
                    a_crow_ptr,
                    a_col_ptr,
                    a_val_ptr,
                    b_crow_ptr,
                    b_col_ptr,
                    b_val_ptr,
                    out_crow_ptr,
                    out_col_ptr,
                    out_val_ptr,
                    alpha_val,
                    nrows);
              }

              // Write results into out tensor in-place.
              TORCH_INTERNAL_ASSERT(out.is_sparse_csr());
              static_cast<SparseCsrTensorImpl*>(out.unsafeGetTensorImpl())
                  ->set_member_tensors(out_crow, out_col, out_val, A.sizes());
            });
      });
}

} // namespace at::native::xpu
