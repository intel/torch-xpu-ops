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
#include <ATen/ceil_div.h>
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
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>
#include <ATen/native/xpu/sycl/LaunchUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/Reduce.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

using namespace at::sparse_csr;
using namespace at::sparse;

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
        const index_t* col_indices_ptr = col_indices.const_data_ptr<index_t>();
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

        const index_t* crow_indices_ptr =
            crow_indices.const_data_ptr<index_t>();
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

template <typename index_t>
inline index_t find_csr_row_for_local_nnz_(
    const index_t* crow_batch,
    int64_t nrows,
    int64_t local_nnz) {
  index_t lo = 0;
  index_t hi = static_cast<index_t>(nrows);
  while (lo < hi) {
    const index_t mid = lo + (hi - lo) / 2;
    if (crow_batch[mid + 1] <= local_nnz) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

inline int64_t choose_work_group_size_(
    int64_t max_work_group_size,
    int64_t k_dim) {
  const int64_t max_work_group_size_ = max_work_group_size;
      //std::min<int64_t>(max_work_group_size, 256);
  const int64_t target_elems_per_thread = 24;
  const int64_t max_pow2_wg =
      at::native::xpu::lastPow2(static_cast<unsigned int>(max_work_group_size_));
  TORCH_INTERNAL_ASSERT(
      max_pow2_wg >= 4,
      "sparse_sampled_addmm_xpu expects work_group_size >= 4, but got max pow2 ",
      max_pow2_wg);
  const int64_t desired_wg =
      std::min<int64_t>(
          at::ceil_div(k_dim, target_elems_per_thread), max_pow2_wg);
  const int64_t rounded_down_wg =
      at::native::xpu::lastPow2(static_cast<unsigned int>(desired_wg));
  const int64_t work_group_size =
      rounded_down_wg < desired_wg ? rounded_down_wg << 1 : rounded_down_wg;
  return std::max<int64_t>(work_group_size, 4);
}

// Extract all tensor metadata (batch + matrix dims) to device tensor.
Tensor compute_batch_sizes_(const Tensor& tensor) {
  const int64_t batch_ndim = std::max<int64_t>(tensor.dim() - 2, 0);
  const int64_t total_ndim = batch_ndim + 2;  // batch dims + 2 matrix dims
  auto meta_cpu = at::empty(
      {total_ndim},
      TensorOptions().dtype(kLong).device(kCPU));
  auto* meta_ptr = meta_cpu.data_ptr<int64_t>();
  for (int64_t d = 0; d < total_ndim; ++d) {
    meta_ptr[d] = tensor.size(d);
  }
  return meta_cpu.to(tensor.device());
}

// Extract all tensor metadata (batch + matrix dims) to device tensor.
Tensor compute_batch_strides_(const Tensor& tensor) {
  const int64_t batch_ndim = std::max<int64_t>(tensor.dim() - 2, 0);
  const int64_t total_ndim = batch_ndim + 2;  // batch dims + 2 matrix dims
  auto meta_cpu = at::empty(
      {total_ndim},
      TensorOptions().dtype(kLong).device(kCPU));
  auto* meta_ptr = meta_cpu.data_ptr<int64_t>();
  for (int64_t d = 0; d < total_ndim; ++d) {
    meta_ptr[d] = tensor.stride(d);
  }
  return meta_cpu.to(tensor.device());
}

template <typename scalar_t, typename index_t>
struct SparseSampledAddmmTreeReduceKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    const int64_t nnz_idx = item.get_group_linear_id();
    const int64_t local_id = item.get_local_id(0);
    const int64_t local_size = item.get_local_range(0);
    const int64_t batch = nnz_idx / nnz_per_batch_;
    const int64_t local_nnz = nnz_idx % nnz_per_batch_;

    if (local_id == 0) {
      init_row_col_(batch, local_nnz);
    } else if (local_id == 1) {
      batch_offsets_[0] =
          compute_batch_base_(batch, mat1_batch_sizes_, mat1_batch_strides_, mat1_batch_ndim_);
    } else if (local_id == 2) {
      batch_offsets_[1] =
          compute_batch_base_(batch, mat2_batch_sizes_, mat2_batch_strides_, mat2_batch_ndim_);
    }
    sycl::group_barrier(item.get_group());

    const index_t row = row_col_[0];
    const index_t col = row_col_[1];
    scalar_t partial = scalar_t(0);
    // Compute matrix strides from metadata vectors (indices: batch_ndim -> matrix dims)
    const int64_t mat1_row_stride = mat1_batch_strides_[mat1_batch_ndim_];
    const int64_t mat1_col_stride = mat1_batch_strides_[mat1_batch_ndim_ + 1];
    const int64_t mat2_row_stride = mat2_batch_strides_[mat2_batch_ndim_];
    const int64_t mat2_col_stride = mat2_batch_strides_[mat2_batch_ndim_ + 1];
    const int64_t mat1_base =
      batch_offsets_[0] + static_cast<int64_t>(row) * mat1_row_stride;
    const int64_t mat2_base =
      batch_offsets_[1] + static_cast<int64_t>(col) * mat2_col_stride;

    for (int64_t kk = local_id; kk < k_dim_; kk += local_size) {
      const int64_t a_offset = mat1_base + kk * mat1_col_stride;
      const int64_t b_offset = mat2_base + kk * mat2_row_stride;
      partial += mat1_[a_offset] * mat2_[b_offset];
    }

    partial_sum_[local_id] = partial;
    sycl::group_barrier(item.get_group());

    for (int64_t stride = local_size >> 1; stride > 0; stride >>= 1) {
      if (local_id < stride) {
        partial_sum_[local_id] += partial_sum_[local_id + stride];
      }
      sycl::group_barrier(item.get_group());
    }

    if (local_id == 0) {
      const int64_t val_idx = batch * val_batch_stride_ + local_nnz;
      const scalar_t current_val = result_values_[val_idx];
      result_values_[val_idx] =
          alpha_val_ * partial_sum_[0] + beta_val_ * current_val;
    }
  }

  SparseSampledAddmmTreeReduceKernelFunctor(
      scalar_t* result_values,
      const index_t* crow_indices,
      const index_t* col_indices,
      const scalar_t* mat1,
      const scalar_t* mat2,
      int64_t nnz_per_batch,
      int64_t nrows,
      int64_t k_dim,
      int64_t crow_batch_stride,
      int64_t col_batch_stride,
      int64_t val_batch_stride,
      const int64_t* mat1_batch_sizes,
      const int64_t* mat1_batch_strides,
      int64_t mat1_batch_ndim,
      const int64_t* mat2_batch_sizes,
      const int64_t* mat2_batch_strides,
      int64_t mat2_batch_ndim,
      scalar_t alpha_val,
      scalar_t beta_val)
      : result_values_(result_values),
        crow_indices_(crow_indices),
        col_indices_(col_indices),
        mat1_(mat1),
        mat2_(mat2),
        nnz_per_batch_(nnz_per_batch),
        nrows_(nrows),
        k_dim_(k_dim),
        crow_batch_stride_(crow_batch_stride),
        col_batch_stride_(col_batch_stride),
        val_batch_stride_(val_batch_stride),
        mat1_batch_sizes_(mat1_batch_sizes),
        mat1_batch_strides_(mat1_batch_strides),
        mat1_batch_ndim_(mat1_batch_ndim),
        mat2_batch_sizes_(mat2_batch_sizes),
        mat2_batch_strides_(mat2_batch_strides),
        mat2_batch_ndim_(mat2_batch_ndim),
        alpha_val_(alpha_val),
        beta_val_(beta_val) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    partial_sum_ = sycl_local_acc_t<scalar_t>(local_size_, cgh);
    row_col_ = sycl_local_acc_t<index_t>(2, cgh);
    batch_offsets_ = sycl_local_acc_t<int64_t>(2, cgh);
  }

  void set_local_size(int64_t local_size) {
    local_size_ = local_size;
  }

  inline void init_row_col_(int64_t batch, int64_t local_nnz) const {
    const index_t* crow_batch = crow_indices_ + batch * crow_batch_stride_;
    const index_t row =
        find_csr_row_for_local_nnz_<index_t>(crow_batch, nrows_, local_nnz);
    row_col_[0] = row;
    row_col_[1] = col_indices_[batch * col_batch_stride_ + local_nnz];
  }

  inline int64_t compute_batch_base_(
      int64_t batch,
      const int64_t* batch_sizes,
      const int64_t* batch_strides,
      int64_t batch_ndim) const {
    int64_t rem = batch;
    int64_t batch_base = 0;
    for (int64_t d = batch_ndim - 1; d >= 0; --d) {
      const int64_t idx = rem % batch_sizes[d];
      rem /= batch_sizes[d];
      batch_base += idx * batch_strides[d];
    }
    return batch_base;
  }

 private:
  scalar_t* result_values_;
  const index_t* crow_indices_;
  const index_t* col_indices_;
  const scalar_t* mat1_;
  const scalar_t* mat2_;
  int64_t nnz_per_batch_;
  int64_t nrows_;
  int64_t k_dim_;
  int64_t crow_batch_stride_;
  int64_t col_batch_stride_;
  int64_t val_batch_stride_;
  const int64_t* mat1_batch_sizes_;
  const int64_t* mat1_batch_strides_;
  int64_t mat1_batch_ndim_;
  const int64_t* mat2_batch_sizes_;
  const int64_t* mat2_batch_strides_;
  int64_t mat2_batch_ndim_;
  scalar_t alpha_val_;
  scalar_t beta_val_;
  int64_t local_size_{1};
  sycl_local_acc_t<scalar_t> partial_sum_;
  sycl_local_acc_t<index_t> row_col_;
  sycl_local_acc_t<int64_t> batch_offsets_;
};

template <typename scalar_t>
void sparse_sampled_addmm_kernel_impl(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  const int64_t k = mat1.size(-1);
  const int64_t m = mat1.size(-2);

  int64_t n_batch = 1;
  for (int64_t d = 0; d < result.dim() - 2; ++d) {
    n_batch *= result.size(d);
  }

  const int64_t nnz_per_batch = result._nnz();
  const int64_t total_nnz = n_batch * nnz_per_batch;

  Tensor crow_indices = result.crow_indices();
  Tensor col_indices = result.col_indices();
  Tensor result_values = result.values();
  const bool has_batch = result.dim() > 2;
  if (has_batch) {
    crow_indices = crow_indices.reshape({n_batch, crow_indices.size(-1)});
    col_indices = col_indices.reshape({n_batch, nnz_per_batch});
    result_values = result_values.reshape({n_batch, nnz_per_batch});
  }

  Tensor mat1_batch_sizes = compute_batch_sizes_(mat1);
  Tensor mat1_batch_strides = compute_batch_strides_(mat1);
  const int64_t mat1_batch_ndim = std::max<int64_t>(mat1.dim() - 2, 0);
  Tensor mat2_batch_sizes = compute_batch_sizes_(mat2);
  Tensor mat2_batch_strides = compute_batch_strides_(mat2);
  const int64_t mat2_batch_ndim = std::max<int64_t>(mat2.dim() - 2, 0);

  auto queue = getCurrentSYCLQueue();
  AT_DISPATCH_INDEX_TYPES(
      crow_indices.scalar_type(),
      "sparse_sampled_addmm_xpu_indices",
      [&] {
        using KernelFn =
            SparseSampledAddmmTreeReduceKernelFunctor<scalar_t, index_t>;

        int64_t max_wg_size = syclMaxWorkGroupSize<KernelFn>();
        int64_t work_group_size = choose_work_group_size_(max_wg_size, k);

        auto kfn = KernelFn(
          result_values.data_ptr<scalar_t>(),
          crow_indices.const_data_ptr<index_t>(),
          col_indices.const_data_ptr<index_t>(),
          mat1.const_data_ptr<scalar_t>(),
          mat2.const_data_ptr<scalar_t>(),
          nnz_per_batch,
          m,
          k,
          has_batch ? crow_indices.stride(0) : 0,
          has_batch ? col_indices.stride(0) : 0,
          has_batch ? result_values.stride(0) : 0,
          mat1_batch_sizes.const_data_ptr<int64_t>(),
          mat1_batch_strides.const_data_ptr<int64_t>(),
          mat1_batch_ndim,
          mat2_batch_sizes.const_data_ptr<int64_t>(),
          mat2_batch_strides.const_data_ptr<int64_t>(),
          mat2_batch_ndim,
          alpha.to<scalar_t>(),
          beta.to<scalar_t>());
        kfn.set_local_size(work_group_size);

        sycl_kernel_submit(
            sycl::range<1>(total_nnz * work_group_size),
            sycl::range<1>(work_group_size),
            queue,
            kfn);
      });
}

void sparse_sampled_addmm_kernel(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kBFloat16, kHalf, result.scalar_type(), "sparse_sampled_addmm_xpu", [&] {
        sparse_sampled_addmm_kernel_impl<scalar_t>(
            self, mat1, mat2, beta, alpha, result);
      });
}

} // namespace at::native::xpu
