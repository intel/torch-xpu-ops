/*
 * Copyright 2020-2026 Intel Corporation
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

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/add.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <oneapi/mkl/spblas.hpp>
#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <ATen/native/sparse/SparseBlasImpl.h>
#include <comm/SYCLContext.h>


namespace at::native::xpu {

using namespace at::sparse_csr;
using namespace at::sparse;

// Map c10 scalar types to oneMKL-compatible types.
// c10::complex<T> and std::complex<T> are layout-identical; reinterpret_cast
// is safe for both pointer and scalar conversions.
template <typename T>
struct get_mkl_type {
  using type = T;
};
template <typename T>
struct get_mkl_type<c10::complex<T>> {
  using type = std::complex<T>;
};

template <typename scalar_t>
struct ReductionAddOp {
  inline scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a + b;
  }
  inline scalar_t identity() const {
    return 0;
  }
  inline scalar_t identity_cpu() const {
    return 0;
  }
};

template <typename scalar_t>
struct ReductionMulOp {
  inline scalar_t operator()(const scalar_t a, const scalar_t b) const {
    return a * b;
  }
  inline scalar_t identity() const {
    return 1;
  }
  inline scalar_t identity_cpu() const {
    return 1;
  }
};

template <
    typename scalar_t,
    typename index_t,
    typename ReductionOp,
    typename acc_t>
struct ReduceSparseCsrDim0KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = item.get_global_linear_id();
    if (tid < new_nnz) {
      index_t col = new_col_indices[tid];
      acc_t v = rop.identity();
      for (int64_t j = 0; j < nnz; j++) {
        if (col == col_indices[j])
          v = rop(v, acc_t(values[j]));
      }
      new_values[tid] = v;
    }
  }

  ReduceSparseCsrDim0KernelFunctor(
      acc_t* new_values,
      const index_t* new_col_indices,
      const int64_t new_nnz,
      const scalar_t* values,
      const index_t* col_indices,
      const int64_t nnz,
      ReductionOp rop)
      : new_values(new_values),
        new_col_indices(new_col_indices),
        new_nnz(new_nnz),
        values(values),
        col_indices(col_indices),
        nnz(nnz),
        rop(rop) {}

 private:
  acc_t* new_values;
  const index_t* new_col_indices;
  const int64_t new_nnz;
  const scalar_t* values;
  const index_t* col_indices;
  const int64_t nnz;
  ReductionOp rop;
};

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim0_xpu_template(
    const Tensor& sparse,
    ReductionOp rop) {
  Tensor col_indices = sparse.col_indices();
  Tensor values = sparse.values();
  auto ncols = sparse.size(1);
  auto nnz = col_indices.numel();

  auto new_col_indices = std::get<0>(at::_unique(col_indices, true, false));
  auto new_nnz = new_col_indices.numel();
  Tensor new_crow_indices =
      at::tensor(ArrayRef<int64_t>{0, new_nnz}, col_indices.options());

  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type(), new_nnz);
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);
  scalar_t* values_ptr = values.data_ptr<scalar_t>();
  acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();
  auto queue = getCurrentSYCLQueue();

  AT_DISPATCH_INDEX_TYPES(
      col_indices.scalar_type(), "reduce_sparse_csr_dim0_xpu_indices", [&]() {
        index_t* col_indices_ptr = col_indices.data_ptr<index_t>();
        index_t* new_col_indices_ptr = new_col_indices.data_ptr<index_t>();
        using KernelFn = ReduceSparseCsrDim0KernelFunctor<
            scalar_t,
            index_t,
            ReductionOp,
            acc_t>;
        int64_t work_group_size = syclMaxWorkGroupSize<KernelFn>();
        int64_t work_group_num =
            (new_nnz + work_group_size - 1) / work_group_size;
        auto kfn = KernelFn(
            new_values_acc_ptr,
            new_col_indices_ptr,
            new_nnz,
            values_ptr,
            col_indices_ptr,
            nnz,
            rop);
        sycl_kernel_submit(
            sycl::range<1>(work_group_num * work_group_size),
            sycl::range<1>(work_group_size),
            queue,
            kfn);
      });
  copy_from_acc_buffer(new_values, new_values_acc);
  return at::native::_sparse_csr_tensor_unsafe(
      new_crow_indices,
      new_col_indices,
      new_values,
      {1, ncols},
      new_values.scalar_type(),
      sparse.layout(),
      new_values.device());
}

template <typename index_t>
struct ReduceCrowIndicesDim1KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t nnz = 0;
    new_crow_indices[0] = 0;
    for (int64_t i = 0; i < nrows; i++) {
      if (crow_indices[i] != crow_indices[i + 1]) {
        row_map[i] = nnz;
        nnz++;
      }
      new_crow_indices[i + 1] = nnz;
    }
  }

  ReduceCrowIndicesDim1KernelFunctor(
      index_t* new_crow_indices,
      index_t* row_map,
      const index_t* crow_indices,
      const int64_t nrows)
      : new_crow_indices(new_crow_indices),
        row_map(row_map),
        crow_indices(crow_indices),
        nrows(nrows) {}

 private:
  index_t* new_crow_indices;
  index_t* row_map;
  const index_t* crow_indices;
  const int64_t nrows;
};

template <
    typename scalar_t,
    typename index_t,
    typename ReductionOp,
    typename acc_t>
struct ReduceSparseCsrDim1KernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = item.get_global_linear_id();
    if (tid < nrows) {
      index_t i_start = crow_indices[tid];
      index_t i_end = crow_indices[tid + 1];
      if (i_start != i_end) {
        acc_t acc = rop.identity();
        for (index_t i = i_start; i < i_end; i++) {
          acc = rop(acc, acc_t(values[i]));
        }
        new_values[row_map[tid]] = acc;
      }
    }
  }

  ReduceSparseCsrDim1KernelFunctor(
      acc_t* new_values,
      const scalar_t* values,
      const index_t* crow_indices,
      const index_t* row_map,
      const int64_t nrows,
      ReductionOp rop)
      : new_values(new_values),
        values(values),
        crow_indices(crow_indices),
        row_map(row_map),
        nrows(nrows),
        rop(rop) {}

 private:
  acc_t* new_values;
  const scalar_t* values;
  const index_t* crow_indices;
  const index_t* row_map;
  const int64_t nrows;
  ReductionOp rop;
};

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim1_xpu_template(
    const Tensor& sparse,
    ReductionOp rop) {
  Tensor crow_indices = sparse.crow_indices();
  auto ioptions = crow_indices.options();
  Tensor values = sparse.values();
  auto nrows = sparse.size(0);

  Tensor new_crow_indices = at::empty({crow_indices.numel()}, ioptions);
  Tensor new_col_indices = at::empty({}, ioptions);
  Tensor row_map = at::empty({nrows}, ioptions);

  using acc_t = at::acc_type<scalar_t, true>;
  auto acc_buffer = at::sparse_csr::create_acc_buffer<acc_t, scalar_t>(
      values.options(), values.scalar_type());
  Tensor new_values = std::get<0>(acc_buffer);
  Tensor new_values_acc = std::get<1>(acc_buffer);
  auto queue = getCurrentSYCLQueue();

  AT_DISPATCH_INDEX_TYPES(
      crow_indices.scalar_type(), "reduce_sparse_csr_dim1_xpu_indices", [&]() {
        using KernelFn = ReduceSparseCsrDim1KernelFunctor<
            scalar_t,
            index_t,
            ReductionOp,
            acc_t>;
        int64_t work_group_size = syclMaxWorkGroupSize<KernelFn>();
        int64_t work_group_num =
            (nrows + work_group_size - 1) / work_group_size;

        index_t* crow_indices_ptr = crow_indices.data_ptr<index_t>();
        index_t* new_crow_indices_ptr = new_crow_indices.data_ptr<index_t>();
        index_t* row_map_ptr = row_map.data_ptr<index_t>();
        ReduceCrowIndicesDim1KernelFunctor<index_t> kfn_crow(
            new_crow_indices_ptr, row_map_ptr, crow_indices_ptr, nrows);
        sycl_kernel_submit(
            sycl::range<1>(1), sycl::range<1>(1), queue, kfn_crow);

        index_t new_nnz = new_crow_indices[-1].item<index_t>();
        new_col_indices.resize_(new_nnz);
        new_col_indices.fill_(index_t(0));
        new_values.resize_(new_nnz);
        new_values_acc.resize_(new_nnz);

        scalar_t* values_ptr = values.data_ptr<scalar_t>();
        acc_t* new_values_acc_ptr = new_values_acc.data_ptr<acc_t>();
        auto kfn = KernelFn(
            new_values_acc_ptr,
            values_ptr,
            crow_indices_ptr,
            row_map_ptr,
            nrows,
            rop);
        sycl_kernel_submit(
            sycl::range<1>(work_group_num * work_group_size),
            sycl::range<1>(work_group_size),
            queue,
            kfn);
      });
  copy_from_acc_buffer(new_values, new_values_acc);
  return at::native::_sparse_csr_tensor_unsafe(
      new_crow_indices,
      new_col_indices,
      new_values,
      {sparse.size(0), 1},
      new_values.scalar_type(),
      sparse.layout(),
      new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_dim01_xpu_template(
    const Tensor& sparse,
    ReductionOp rop) {
  auto ioptions = sparse.col_indices().options();
  Tensor values = sparse.values();
  auto numel = values.numel();
  auto nnz = std::min<int64_t>(1, numel);

  auto result_dtype =
      at::isIntegralType(values.scalar_type(), /*includeBool=*/true)
      ? ScalarType::Long
      : values.scalar_type();
  Tensor new_values, new_values_acc;
  if (numel > 0) {
    new_values = at::empty({1}, values.options().dtype(result_dtype));
    new_values_acc = at::empty({1}, values.options());
    auto iter = TensorIterator::reduce_op(new_values_acc, values);
    gpu_reduce_kernel<scalar_t, scalar_t>(
        iter, func_wrapper<scalar_t>(rop), rop.identity_cpu());
    new_values.copy_(new_values_acc);
  } else {
    new_values = at::empty({}, values.options().dtype(result_dtype));
  }
  Tensor new_col_indices = at::zeros({nnz}, ioptions);
  Tensor new_crow_indices = at::tensor(ArrayRef<int64_t>{0, nnz}, ioptions);
  return at::native::_sparse_csr_tensor_unsafe(
      new_crow_indices,
      new_col_indices,
      new_values,
      {1, std::min<int64_t>(1, sparse.size(1))},
      new_values.scalar_type(),
      sparse.layout(),
      new_values.device());
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_xpu_template(
    const Tensor& sparse,
    std::vector<int64_t> dims,
    ReductionOp rop) {
  if (dims.size() == 1) {
    if (dims[0] == 0) {
      return reduce_sparse_csr_dim0_xpu_template<scalar_t>(sparse, rop);
    } else {
      TORCH_INTERNAL_ASSERT(dims[0] == 1);
      return reduce_sparse_csr_dim1_xpu_template<scalar_t>(sparse, rop);
    }
  } else if (dims.size() == 2) {
    TORCH_INTERNAL_ASSERT(
        ((dims[0] == 0 && dims[1] == 1) || (dims[0] == 1 && dims[1] == 0)));
    return reduce_sparse_csr_dim01_xpu_template<scalar_t>(sparse, rop);
  }
  TORCH_INTERNAL_ASSERT(dims.size() == 0);
  return sparse.clone();
}

template <typename scalar_t, typename ReductionOp>
Tensor reduce_sparse_csr_xpu_template(
    const Tensor& sparse,
    IntArrayRef dims_to_sum,
    bool keepdim,
    ReductionOp rop) {
  TORCH_INTERNAL_ASSERT(sparse.is_sparse_csr());
  TORCH_CHECK(
      keepdim,
      "reduction operations on CSR tensors with keepdim=False is unsupported");
  TORCH_INTERNAL_ASSERT(sparse.is_xpu());

  const int64_t input_dim = sparse.dim();
  TORCH_INTERNAL_ASSERT(input_dim == 2);
  auto dims = dims_to_sum.vec();
  maybe_wrap_dims(dims, input_dim);
  if (dims.size() == 0) {
    dims.emplace_back(0);
    dims.emplace_back(1);
  }
  return reduce_sparse_csr_xpu_template<scalar_t>(sparse, dims, rop);
}

Tensor _sparse_csr_sum_xpu_kernel(
    const Tensor& input,
    IntArrayRef dims_to_sum,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = at::sparse_csr::to_type(input, dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_sum_xpu", [&] {
        using acc_t = at::acc_type<scalar_t, true>;
        result = reduce_sparse_csr_xpu_template<scalar_t>(
            input_, dims_to_sum, keepdim, ReductionAddOp<acc_t>());
      });
  return result;
}

Tensor _sparse_csr_prod_xpu_kernel(
    const Tensor& input,
    IntArrayRef dims_to_reduce,
    bool keepdim,
    std::optional<ScalarType> dtype) {
  ScalarType dtype_ = dtype.value_or(input.scalar_type());
  Tensor input_ = input.to(dtype_);
  Tensor result;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "_sparse_csr_prod_xpu", [&] {
        result = reduce_sparse_csr_xpu_template<scalar_t>(
            input_, dims_to_reduce, keepdim, ReductionMulOp<scalar_t>());
      });
  return result;
}

template <typename input_t, typename output_t>
struct ConvertIndicesFromCooToCsrXPUFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto linear_id = item.get_global_linear_id();
    if (linear_id == 0) {
      for (int64_t i = 0; i <= data_in_[0]; i++)
        data_out_[i] = static_cast<output_t>(0);
    } else if (linear_id < numel_) {
      for (int64_t i = data_in_[linear_id - 1]; i < data_in_[linear_id]; i++)
        data_out_[i + 1] = static_cast<output_t>(linear_id);
    } else if (linear_id == numel_) {
      for (int64_t i = data_in_[numel_ - 1] + 1; i < size_ + 1; i++)
        data_out_[i] = static_cast<output_t>(numel_);
    }
  }
  ConvertIndicesFromCooToCsrXPUFunctor(
      int64_t numel,
      const input_t* data_in,
      output_t* data_out,
      const int64_t size)
      : numel_(numel), data_in_(data_in), data_out_(data_out), size_(size) {}

 private:
  int64_t numel_;
  const input_t* data_in_;
  output_t* data_out_;
  const int64_t size_;
};

template <typename input_t, typename output_t>
struct ConvertIndicesFromCsrToCooXPUFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t tid = item.get_global_linear_id();
    if (tid < nrows_ * nbatches_) {
      int64_t b = tid / nrows_;
      int64_t i_ = b * (nrows_ + 1) + tid % nrows_;
      for (int64_t i = data_in_[i_]; i < data_in_[i_ + 1]; i++) {
        data_out_[b * nnz_ + i] = static_cast<output_t>(tid % nrows_);
      }
    }
  }
  ConvertIndicesFromCsrToCooXPUFunctor(
      output_t* data_out,
      const input_t* data_in,
      const int64_t nrows,
      const int64_t nnz,
      const int64_t nbatches)
      : data_out_(data_out),
        data_in_(data_in),
        nrows_(nrows),
        nnz_(nnz),
        nbatches_(nbatches) {}

 private:
  output_t* data_out_;
  const input_t* data_in_;
  const int64_t nrows_;
  const int64_t nnz_;
  const int64_t nbatches_;
};

template <typename input_t, typename output_t>
void launch_convert_indices_from_coo_to_csr_xpu_kernel(
    const Tensor& result,
    const Tensor& input,
    const int64_t size) {
  int64_t numel = input.numel();
  if (numel == 0) {
    result.zero_();
    return;
  }

  const input_t* data_in = input.const_data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  auto functor = ConvertIndicesFromCooToCsrXPUFunctor<input_t, output_t>(
      numel, data_in, data_out, size);

  int64_t wgroup_size = syclMaxWorkGroupSize(functor);
  int64_t ngroups = (numel + wgroup_size - 1) / wgroup_size;
  sycl::range<1> global_range(ngroups * wgroup_size);
  sycl::range<1> local_range(wgroup_size);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), functor);
}

template <typename input_t, typename output_t>
void launch_convert_indices_from_csr_to_coo_xpu_kernel(
    const Tensor& indices,
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool transpose = false) {
  int64_t nrows = crow_indices.size(-1) - 1;
  int64_t nnz = col_indices.size(-1);
  if (nrows == 0 || nnz == 0) {
    indices.zero_();
    return;
  }
  int64_t total_nnz = col_indices.numel();
  int64_t batch_ndim = crow_indices.dim() - 1;
  if (batch_ndim > 0) {
    auto batch_indices = indices.narrow(0, 0, batch_ndim);
    batch_indices.copy_(
        at::sparse::full_coo_indices(
            crow_indices.sizes().slice(0, batch_ndim), indices.options())
            .repeat_interleave(nnz, 1));
  }

  auto crow_indices_ = crow_indices.expect_contiguous();
  const input_t* crow_indices_data_in =
      crow_indices_->const_data_ptr<input_t>();
  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  auto row0 = indices.select(0, transpose ? batch_ndim + 1 : batch_ndim + 0);
  auto row1 = indices.select(0, transpose ? batch_ndim + 0 : batch_ndim + 1);
  auto col_indices_ = col_indices.expect_contiguous();
  row1.copy_(col_indices_->view({-1}));
  output_t* data_out = row0.data_ptr<output_t>();

  // Run nrows * nbatches threads...
  int64_t nbatches = total_nnz / nnz;
  auto functor = ConvertIndicesFromCsrToCooXPUFunctor<input_t, output_t>(
      data_out, crow_indices_data_in, nrows, nnz, nbatches);

  int64_t THREADS = syclMaxWorkGroupSize(functor);
  int64_t GROUPS = (nrows * nbatches + THREADS) / THREADS;

  sycl::range<1> global_range(GROUPS * THREADS);
  sycl::range<1> local_range(THREADS);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), functor);
}

void convert_indices_from_coo_to_csr_structured_kernel(
    const Tensor& input,
    const int64_t size,
    const bool out_int32,
    const Tensor& result) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_xpu", [&] {
          launch_convert_indices_from_coo_to_csr_xpu_kernel<scalar_t, int>(
              result, input, size);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "convert_indices_from_coo_to_csr_xpu", [&] {
          launch_convert_indices_from_coo_to_csr_xpu_kernel<scalar_t, int64_t>(
              result, input, size);
        });
  }
}

void convert_indices_from_csr_to_coo_structured_kernel(
    const Tensor& crow_indices,
    const Tensor& col_indices,
    const bool out_int32,
    const bool transpose,
    const Tensor& result) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_xpu", [&] {
          launch_convert_indices_from_csr_to_coo_xpu_kernel<scalar_t, int>(
              result, crow_indices, col_indices, transpose);
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(
        crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_xpu", [&] {
          launch_convert_indices_from_csr_to_coo_xpu_kernel<scalar_t, int64_t>(
              result, crow_indices, col_indices, transpose);
        });
  }
}

namespace {

namespace mkl_sp = oneapi::mkl::sparse;

// Invoke oneMKL sparse::gemm for a CSR matrix:
//   Y = alpha * op(A_csr) @ X + beta * Y
template <typename mkl_scalar_t, typename index_t>
void onemkl_csr_gemm_impl(
    sycl::queue& queue,
    int64_t m,
    int64_t k,
    int64_t n,
    int64_t nnz,
    mkl_scalar_t alpha,
    index_t* crow_ptr,
    index_t* col_ind,
    mkl_scalar_t* values,
    mkl_scalar_t* x_ptr,
    int64_t ldx,
    mkl_scalar_t beta,
    mkl_scalar_t* y_ptr,
    int64_t ldy) {
  mkl_sp::matrix_handle_t handle = nullptr;
  mkl_sp::init_matrix_handle(&handle);
  mkl_sp::set_csr_data(
      queue, handle, m, k, nnz,
      oneapi::mkl::index_base::zero,
      crow_ptr, col_ind, values);
  mkl_sp::optimize_gemm(
      queue,
      oneapi::mkl::layout::row_major,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      handle, n);
  mkl_sp::gemm(
      queue,
      oneapi::mkl::layout::row_major,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      alpha, handle,
      x_ptr, n, ldx,
      beta, y_ptr, ldy);
  mkl_sp::release_matrix_handle(queue, &handle);
}

// result = alpha * sparse @ x + beta * result
// sparse must be 2D CSR or BSR; x must be 2D contiguous dense.
// result must already hold the bias before entry (beta scales it in-place).
// half/bfloat16 are promoted to float32 (oneMKL constraint).
void onemkl_sparse_gemm(
    const Tensor& sparse,
    const Tensor& x,
    const Scalar& alpha,
    const Scalar& beta,
    Tensor& result) {
  auto queue = getCurrentSYCLQueue();

  const auto compute_dtype =
      (result.scalar_type() == kHalf || result.scalar_type() == kBFloat16)
      ? kFloat
      : result.scalar_type();
  const bool needs_promotion = (compute_dtype != result.scalar_type());

  const Tensor sparse_c = needs_promotion ? sparse.to(compute_dtype) : sparse;
  const Tensor x_c = needs_promotion
      ? x.contiguous().to(compute_dtype)
      : x.contiguous();
  // Always compute into a contiguous row-major buffer so that oneMKL can use
  // ldy = n.  If result is column-major (transposed out) or promoted, copy back.
  const bool result_non_contig = !result.is_contiguous();
  Tensor result_buf = (needs_promotion || result_non_contig)
      ? result.contiguous().to(compute_dtype)
      : result;

  const int64_t m = sparse_c.size(0);
  const int64_t k = sparse_c.size(1);
  const int64_t n = x_c.size(1);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      compute_dtype, "onemkl_sparse_gemm", [&]() {
        using mkl_t = typename get_mkl_type<scalar_t>::type;

        const scalar_t alpha_c10 = alpha.to<scalar_t>();
        const scalar_t beta_c10  = beta.to<scalar_t>();
        const mkl_t alpha_mkl = *reinterpret_cast<const mkl_t*>(&alpha_c10);
        const mkl_t beta_mkl  = *reinterpret_cast<const mkl_t*>(&beta_c10);

        mkl_t* x_ptr = reinterpret_cast<mkl_t*>(x_c.data_ptr<scalar_t>());
        mkl_t* y_ptr = reinterpret_cast<mkl_t*>(result_buf.data_ptr<scalar_t>());

        AT_DISPATCH_INDEX_TYPES(
            sparse_c.crow_indices().scalar_type(),
            "onemkl_sparse_gemm_idx",
            [&]() {
              index_t* crow_ptr =
                  sparse_c.crow_indices().data_ptr<index_t>();
              index_t* col_ptr =
                  sparse_c.col_indices().data_ptr<index_t>();

              TORCH_INTERNAL_ASSERT(
                  sparse_c.layout() == kSparseCsr,
                  "onemkl_sparse_gemm: expected CSR layout");
              const int64_t nnz = sparse_c.values().numel();
              Tensor vals_c = sparse_c.values().contiguous();
              mkl_t* val_ptr =
                  reinterpret_cast<mkl_t*>(vals_c.data_ptr<scalar_t>());
              onemkl_csr_gemm_impl<mkl_t, index_t>(
                  queue, m, k, n, nnz,
                  alpha_mkl, crow_ptr, col_ptr, val_ptr,
                  x_ptr, n, beta_mkl, y_ptr, n);
            });
      });

  if (needs_promotion || result_non_contig) {
    result.copy_(result_buf);
  }
}

// ---------------------------------------------------------------------------
// Sparse @ sparse → sparse using oneMKL sparse::matmat
//   Computes result = alpha * A_csr @ B_csr + beta * old_result
//   (caller must have already copied input into result when beta != 0)
// ---------------------------------------------------------------------------
template <typename scalar_t, typename index_t>
// input  = bias term (M); used only when beta != 0.
// result = output tensor (sparse CSR); its internal members are replaced in-place.
// A, B   = CSR operands for the product A @ B.
void onemkl_spgemm_impl(
    sycl::queue& queue,
    const Tensor& A,
    const Tensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& input,
    const Tensor& result) {
  using mkl_t = typename get_mkl_type<scalar_t>::type;
  namespace sp = oneapi::mkl::sparse;
  using mkl_ib = oneapi::mkl::index_base;
  using mkl_tr = oneapi::mkl::transpose;

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  const int64_t n = B.size(1);

  // Clone A and B CSR arrays into fresh independent buffers so that oneMKL
  // (or the XPU runtime) cannot corrupt the caller's input tensors even if it
  // temporarily reuses those device-memory regions as internal workspace.
  Tensor a_crow = A.crow_indices().clone();
  Tensor a_col  = A.col_indices().clone();
  Tensor a_vals = A.values().contiguous().clone();
  Tensor b_crow = B.crow_indices().clone();
  Tensor b_col  = B.col_indices().clone();
  Tensor b_vals = B.values().contiguous().clone();

  // Allocate crow for C; use placeholder col/val (1 element) for initial handle
  auto idx_opts = a_crow.options();
  auto val_opts = a_vals.options();
  Tensor c_crow = at::empty({m + 1}, idx_opts);
  // Allocate placeholder col/val for the initial hC registration.
  // Some oneMKL versions write column indices into the placeholder buffer
  // during compute_structure (device-side), so allocate the MAXIMUM POSSIBLE
  // nnz (= m * n for dense inputs) to avoid any buffer overflow.
  const int64_t max_c_nnz = m * n;
  Tensor c_col_ph = at::empty({max_c_nnz}, idx_opts);
  Tensor c_val_ph = at::empty({max_c_nnz}, val_opts);

  sp::matrix_handle_t hA = nullptr, hB = nullptr, hC = nullptr;
  sp::init_matrix_handle(&hA);
  sp::init_matrix_handle(&hB);
  sp::init_matrix_handle(&hC);

  index_t* a_crow_ptr = a_crow.data_ptr<index_t>();
  index_t* a_col_ptr  = a_col.data_ptr<index_t>();
  mkl_t*   a_val_ptr  = reinterpret_cast<mkl_t*>(a_vals.data_ptr<scalar_t>());
  index_t* b_crow_ptr = b_crow.data_ptr<index_t>();
  index_t* b_col_ptr  = b_col.data_ptr<index_t>();
  mkl_t*   b_val_ptr  = reinterpret_cast<mkl_t*>(b_vals.data_ptr<scalar_t>());

  sp::set_csr_data(queue, hA, m, k, A._nnz(), mkl_ib::zero,
                   a_crow_ptr, a_col_ptr, a_val_ptr);
  sp::set_csr_data(queue, hB, k, n, B._nnz(), mkl_ib::zero,
                   b_crow_ptr, b_col_ptr, b_val_ptr);
  sp::set_csr_data(queue, hC, m, n, 0, mkl_ib::zero,
                   c_crow.data_ptr<index_t>(),
                   c_col_ph.data_ptr<index_t>(),
                   reinterpret_cast<mkl_t*>(c_val_ph.data_ptr<scalar_t>()));

  sp::matmat_descr_t desc = nullptr;
  sp::init_matmat_descr(&desc);
  sp::set_matmat_data(
      desc,
      sp::matrix_view_descr::general, mkl_tr::nontrans,
      sp::matrix_view_descr::general, mkl_tr::nontrans,
      sp::matrix_view_descr::general);

  // Step 1: work estimation
  std::int64_t work_sz = 0;
  sp::matmat(queue, hA, hB, hC,
             sp::matmat_request::get_work_estimation_buf_size,
             desc, &work_sz, nullptr, {}).wait();
  void* work_buf = work_sz ? sycl::malloc_device(work_sz, queue) : nullptr;
  sp::matmat(queue, hA, hB, hC,
             sp::matmat_request::work_estimation,
             desc, &work_sz, work_buf, {}).wait();

  // Step 2: compute sparsity structure of C
  std::int64_t str_sz = 0;
  sp::matmat(queue, hA, hB, hC,
             sp::matmat_request::get_compute_structure_buf_size,
             desc, &str_sz, nullptr, {}).wait();
  void* str_buf = str_sz ? sycl::malloc_device(str_sz, queue) : nullptr;
  sp::matmat(queue, hA, hB, hC,
             sp::matmat_request::compute_structure,
             desc, &str_sz, str_buf, {}).wait();
  sp::matmat(queue, hA, hB, hC,
             sp::matmat_request::finalize_structure,
             desc, &str_sz, str_buf, {}).wait();

  // c_crow[m] now holds nnz of C (synchronous .item() copies to host)
  int64_t c_nnz = c_crow[m].item<int64_t>();

  // Allocate col and val for C
  Tensor c_col = at::empty({c_nnz}, idx_opts);
  Tensor c_val = at::empty({c_nnz}, val_opts);

  // Re-register hC with full arrays
  sp::set_csr_data(queue, hC, m, n, c_nnz, mkl_ib::zero,
                   c_crow.data_ptr<index_t>(),
                   c_col.data_ptr<index_t>(),
                   reinterpret_cast<mkl_t*>(c_val.data_ptr<scalar_t>()));

  // Step 3: compute values of C
  std::int64_t cmp_sz = 0;
  sp::matmat(queue, hA, hB, hC,
             sp::matmat_request::get_compute_buf_size,
             desc, &cmp_sz, nullptr, {}).wait();
  void* cmp_buf = cmp_sz ? sycl::malloc_device(cmp_sz, queue) : nullptr;
  sp::matmat(queue, hA, hB, hC,
             sp::matmat_request::compute,
             desc, &cmp_sz, cmp_buf, {}).wait();
  sp::matmat(queue, hA, hB, hC,
             sp::matmat_request::finalize,
             desc, &cmp_sz, cmp_buf, {}).wait();

  if (work_buf) sycl::free(work_buf, queue);
  if (str_buf)  sycl::free(str_buf, queue);
  if (cmp_buf)  sycl::free(cmp_buf, queue);

  sp::release_matmat_descr(&desc);
  sp::release_matrix_handle(queue, &hA);
  sp::release_matrix_handle(queue, &hB);
  sp::release_matrix_handle(queue, &hC);
  // release_matrix_handle is asynchronous: wait for all device-side handle
  // teardown to complete before a_vals/b_vals go out of scope and their
  // device memory is freed.
  queue.wait();

  // c_crow/c_col/c_val now hold T = A @ B.  Apply alpha.
  scalar_t alpha_val = alpha.to<scalar_t>();
  c_val.mul_(alpha_val);

  // Use set_member_tensors to directly replace the SparseCsrTensorImpl's
  // internal tensors.  The .col_indices() / .values() accessors return an
  // .alias() with a DIFFERENT TensorImpl, so resize_+copy_ on those aliases
  // does NOT propagate back into the SparseCsrTensorImpl when it requires
  // storage growth.  set_member_tensors replaces the internal tensors in-place.
  auto* result_impl = get_sparse_csr_impl(result);

  scalar_t beta_val = beta.to<scalar_t>();
  if (beta_val == scalar_t(0)) {
    // result = alpha * T — replace result's internal tensors with T
    result_impl->set_member_tensors(c_crow, c_col, c_val, result.sizes());
  } else {
    // result = alpha * T + beta * input
    // Build T = alpha*(A@B) as a temporary sparse CSR tensor, then add
    // beta * input.  We use the `input` parameter directly — never read
    // `result` here, since result may still be an empty/uninitialized tensor
    // when the caller hasn't done a pre-copy (sparse copy_ requires equal nnz).
    Tensor T = at::native::_sparse_csr_tensor_unsafe(
        c_crow, c_col, c_val,
        {m, n}, c_val.scalar_type(), kSparseCsr, c_val.device());
    Tensor new_result = at::add(T, input, Scalar(beta_val));
    result_impl->set_member_tensors(
        new_result.crow_indices(),
        new_result.col_indices(),
        new_result.values(),
        new_result.sizes());
  }
  // Ensure all asynchronous XPU kernels (mul_, at::add) have completed
  // before c_val, T, and other local tensors go out of scope and their
  // device memory is freed back to the allocator.
  queue.wait();
}

// Dispatcher: selects scalar_t and index_t then calls onemkl_spgemm_impl.
// A and B must be CSR.  input is the bias term M (used only when beta != 0).
// result is the output tensor (CSR); its existing contents are IGNORED —
// the caller must NOT pre-copy input into result.
// half/bfloat16 are promoted to float32 for the computation.
void onemkl_spgemm(
    const Tensor& A,
    const Tensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& input,
    const Tensor& result) {
  auto queue = getCurrentSYCLQueue();
  const auto dt = result.scalar_type();
  const bool needs_promo = (dt == kHalf || dt == kBFloat16);

  if (needs_promo) {
    // Promote A, B, input to float32; compute in float32 then downcast.
    Tensor a_f32     = A.to(kFloat);
    Tensor b_f32     = B.to(kFloat);
    Tensor input_f32 = input.to(kFloat);
    // Empty float32 CSR placeholder; set_member_tensors will fill it.
    Tensor result_f32 = at::zeros(
        {A.size(0), B.size(1)},
        a_f32.options().layout(kSparseCsr));

    AT_DISPATCH_INDEX_TYPES(
        a_f32.crow_indices().scalar_type(), "onemkl_spgemm_promo_idx", [&]() {
          onemkl_spgemm_impl<float, index_t>(
              queue, a_f32, b_f32, beta, alpha, input_f32, result_f32);
        });

    // Downcast float32 values to half/bfloat16 and install into result.
    get_sparse_csr_impl(result)->set_member_tensors(
        result_f32.crow_indices(),
        result_f32.col_indices(),
        result_f32.values().to(dt),
        result.sizes());
    return;
  }

  // sparse::matmat does not support complex types; fall back to dense mm.
  const bool is_complex = (dt == kComplexFloat || dt == kComplexDouble);
  if (is_complex) {
    Tensor T_dense = at::mm(A.to_dense(), B.to_dense());
    T_dense.mul_(alpha);
    Tensor T_csr = T_dense.to_sparse_csr();
    auto* result_impl = get_sparse_csr_impl(result);
    const double beta_val = beta.toComplexDouble().real();
    if (beta_val == 0.) {
      result_impl->set_member_tensors(
          T_csr.crow_indices(), T_csr.col_indices(), T_csr.values(),
          result.sizes());
    } else {
      Tensor new_result = at::add(T_csr, input, beta);
      result_impl->set_member_tensors(
          new_result.crow_indices(), new_result.col_indices(),
          new_result.values(), new_result.sizes());
    }
    return;
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      dt, "onemkl_spgemm", [&]() {
        AT_DISPATCH_INDEX_TYPES(
            A.crow_indices().scalar_type(), "onemkl_spgemm_idx", [&]() {
              onemkl_spgemm_impl<scalar_t, index_t>(
                  queue, A, B, beta, alpha, input, result);
            });
      });
}

} // anonymous namespace

void addmm_out_sparse_csr(
    const Tensor& input,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_INTERNAL_ASSERT(
      !((mat1.layout() == kStrided) && (mat2.layout() == kStrided) &&
        (result.layout() == kStrided)),
      "Expected at least one sparse input");

  using at::native::sparse::impl::_compressed_row_strided_addmm_out;

  // Layout checks are nested: mat1 (outer) → mat2 → result (inner), following
  // the same structure as the CUDA implementation in SparseBlasImpl.cpp.
  // Conditions are ordered: bsr/bsc first (manage input→result copy themselves),
  // then strided, csr, csc.  Valid combinations terminate in a return; invalid
  // ones fall through to the TORCH_CHECK below.

  // ---- mat1 = BSR ----------------------------------------------------------
  // oneMKL sparse::gemm does not support BSR on GPU; use the reference
  // implementation which is block-aware and float64-accurate.
  // _compressed_row_strided_addmm_out copies input into result internally.
  if (mat1.layout() == kSparseBsr) {
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        _compressed_row_strided_addmm_out(input, mat1, mat2, beta, alpha, result);
        return;
      }
    }
  }

  // ---- mat1 = dense (strided) ----------------------------------------------
  if (mat1.layout() == kStrided) {
    // dense @ CSR → transpose trick: compute CSR.T @ mat1.T, copy back
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kStrided) {
        Tensor sparse_t = mat2.transpose(-2, -1).to_sparse_csr(); // CSR.T → CSC → CSR
        Tensor result_T_buf = input.transpose(-2, -1).contiguous().clone();
        onemkl_sparse_gemm(
            sparse_t,
            mat1.transpose(-2, -1).contiguous(),
            alpha, beta, result_T_buf);
        result.copy_(result_T_buf.transpose(-2, -1));
        return;
      }
    }
    // dense @ CSC → transpose trick: CSC.T is a CSR view, oneMKL handles directly
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kStrided) {
        Tensor sparse_t = mat2.transpose(-2, -1); // CSC.T = CSR
        Tensor result_T_buf = input.transpose(-2, -1).contiguous().clone();
        onemkl_sparse_gemm(
            sparse_t,
            mat1.transpose(-2, -1).contiguous(),
            alpha, beta, result_T_buf);
        result.copy_(result_T_buf.transpose(-2, -1));
        return;
      }
    }
    // dense @ BSC → transpose trick + _compressed_row_strided_addmm_out
    // Compute (BSC.T=BSR) @ mat1.T = result.T, then copy back.
    if (mat2.layout() == kSparseBsc) {
      if (result.layout() == kStrided) {
        Tensor a_bsr   = mat2.transpose(-2, -1);            // BSC.T = BSR
        Tensor b_dense = mat1.transpose(-2, -1).contiguous();
        Tensor c_dense = input.transpose(-2, -1).contiguous();
        Tensor result_T_buf = at::empty_like(c_dense);
        _compressed_row_strided_addmm_out(
            c_dense, a_bsr, b_dense, beta, alpha, result_T_buf);
        result.copy_(result_T_buf.transpose(-2, -1));
        return;
      }
    }
  }

  // ---- mat1 = CSR ----------------------------------------------------------
  if (mat1.layout() == kSparseCsr) {
    // CSR @ dense → oneMKL sparse::gemm
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        if (!result.is_same(input)) {
          result.copy_(input);
        }
        onemkl_sparse_gemm(mat1, mat2, alpha, beta, result);
        return;
      }
    }
    // CSR @ CSR → CSR
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kSparseCsr) {
        onemkl_spgemm(mat1, mat2, beta, alpha, input, result);
        return;
      }
    }
    // CSR @ CSC → CSR  (convert mat2 CSC → CSR)
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kSparseCsr) {
        onemkl_spgemm(mat1, mat2.to_sparse_csr(), beta, alpha, input, result);
        return;
      }
    }
  }

  // ---- mat1 = CSC ----------------------------------------------------------
  if (mat1.layout() == kSparseCsc) {
    // CSC @ dense → to_sparse_csr + oneMKL sparse::gemm
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        if (!result.is_same(input)) {
          result.copy_(input);
        }
        onemkl_sparse_gemm(mat1.to_sparse_csr(), mat2, alpha, beta, result);
        return;
      }
    }
    // CSC @ CSR → CSR  (convert mat1 CSC → CSR)
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kSparseCsr) {
        onemkl_spgemm(mat1.to_sparse_csr(), mat2, beta, alpha, input, result);
        return;
      }
    }
    if (mat2.layout() == kSparseCsc) {
      // CSC @ CSC → CSR  (convert both to CSR)
      if (result.layout() == kSparseCsr) {
        onemkl_spgemm(
            mat1.to_sparse_csr(), mat2.to_sparse_csr(), beta, alpha, input, result);
        return;
      }
      // CSC @ CSC → CSC: reinterpret CSC storage directly as the CSR of the
      // transpose and compute B^T[n×k] @ A^T[k×m] = C^T[n×m] in CSR space.
      // This avoids to_sparse_csr()/to_sparse_csc() roundtrips that can produce
      // unsorted column indices which confuse oneMKL.
      //   A_csc[m×k]   ≡  A^T  as CSR[k×m]  (ccol→crow, row→col, val→val)
      //   B_csc[k×n]   ≡  B^T  as CSR[n×k]
      //   inp_csc[m×n] ≡  inp^T as CSR[n×m]
      //   result of B^T@A^T in CSR[n×m] ≡ result CSC[m×n] (crow→ccol, col→row)
      if (result.layout() == kSparseCsc) {
        const int64_t m = mat1.size(0), k = mat1.size(1), n = mat2.size(1);
        Tensor a_T = at::native::_sparse_csr_tensor_unsafe(
            mat1.ccol_indices().clone(), mat1.row_indices().clone(),
            mat1.values().clone(), {k, m},
            mat1.scalar_type(), kSparseCsr, mat1.device());
        Tensor b_T = at::native::_sparse_csr_tensor_unsafe(
            mat2.ccol_indices().clone(), mat2.row_indices().clone(),
            mat2.values().clone(), {n, k},
            mat2.scalar_type(), kSparseCsr, mat2.device());
        Tensor inp_T = at::native::_sparse_csr_tensor_unsafe(
            input.ccol_indices().clone(), input.row_indices().clone(),
            input.values().clone(), {n, m},
            input.scalar_type(), kSparseCsr, input.device());
        Tensor result_T = at::zeros({n, m}, a_T.options().layout(kSparseCsr));
        onemkl_spgemm(b_T, a_T, beta, alpha, inp_T, result_T);
        // result_T CSR[n×m]: crow_indices→ccol_indices, col_indices→row_indices
        get_sparse_csr_impl(result)->set_member_tensors(
            result_T.crow_indices(),
            result_T.col_indices(),
            result_T.values(),
            result.sizes());
        return;
      }
    }
  }

  TORCH_CHECK(
      false,
      "addmm: computation on XPU is not implemented for ",
      result.layout(), " + ", mat1.layout(), " @ ", mat2.layout());
}

} // namespace at::native::xpu
