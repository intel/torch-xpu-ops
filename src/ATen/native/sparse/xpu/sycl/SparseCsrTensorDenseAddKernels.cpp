/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorAddKernels.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t, typename index_t>
struct DenseSparseCompressedAddFunctor {
  void operator()(sycl::item<1> item) const {
    const auto task = static_cast<int64_t>(item.get_id(0));
    const auto batch = task / nnz_;
    const auto i = task % nnz_;
    const auto indices_offset = batch * (compressed_size_ + 1);
    int64_t lower = 0;
    int64_t upper = compressed_size_;
    while (lower < upper) {
      const auto middle = lower + (upper - lower) / 2;
      if (compressed_indices_[indices_offset + middle + 1] <= i) {
        lower = middle + 1;
      } else {
        upper = middle;
      }
    }
    const auto plain = plain_indices_[task];
    const auto row = is_csr_ ? lower : plain;
    const auto col = is_csr_ ? plain : lower;
    output_[(batch * rows_ + row) * cols_ + col] += alpha_ * values_[task];
  }

  scalar_t* output_;
  const index_t* compressed_indices_;
  const index_t* plain_indices_;
  const scalar_t* values_;
  scalar_t alpha_;
  int64_t rows_;
  int64_t cols_;
  int64_t compressed_size_;
  int64_t nnz_;
  bool is_csr_;
};

void add_out_dense_sparse_compressed_kernel(
    Tensor& output,
    const Tensor& sparse,
    const Scalar& alpha) {
  if (sparse._nnz() == 0 || output.numel() == 0) {
    return;
  }

  TORCH_INTERNAL_ASSERT(output.is_contiguous());
  const auto rows = output.size(-2);
  const auto cols = output.size(-1);
  const auto batches = output.numel() / (rows * cols);
  const auto nnz = sparse._nnz();
  const bool is_csr = sparse.layout() == kSparseCsr;
  const auto compressed_size = is_csr ? rows : cols;
  Tensor compressed_indices;
  Tensor plain_indices;
  std::tie(compressed_indices, plain_indices) =
      at::sparse_csr::getCompressedPlainIndices(sparse);
  compressed_indices =
      compressed_indices.reshape({batches, compressed_size + 1}).contiguous();
  plain_indices = plain_indices.reshape({batches, nnz}).contiguous();
  const auto values = sparse.values()
                          .to(output.scalar_type())
                          .reshape({batches, nnz})
                          .contiguous();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      output.scalar_type(),
      "add_out_dense_sparse_compressed_xpu",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            compressed_indices.scalar_type(),
            "add_out_dense_sparse_compressed_xpu_index",
            [&] {
              const auto functor =
                  DenseSparseCompressedAddFunctor<scalar_t, index_t>{
                      output.data_ptr<scalar_t>(),
                      compressed_indices.const_data_ptr<index_t>(),
                      plain_indices.const_data_ptr<index_t>(),
                      values.const_data_ptr<scalar_t>(),
                      alpha.to<scalar_t>(),
                      rows,
                      cols,
                      compressed_size,
                      nnz,
                      is_csr};
              sycl_kernel_submit(
                  sycl::range<1>(batches * nnz),
                  getCurrentSYCLQueue(),
                  functor);
            });
      });
}

} // namespace at::native::xpu
