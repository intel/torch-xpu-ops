/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>
#include <ATen/native/xpu/mkl/TorchToMklType.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/zeros_like.h>

#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <oneapi/mkl/lapack.hpp>

namespace at::native::xpu {

#define SYCL_ONEMKL_SUBMIT(q, routine, ...) \
  {                                         \
    auto e = (routine(__VA_ARGS__));        \
    (q).throw_asynchronous();               \
  }

// Transforms TransposeType into the BLAS / LAPACK format
static oneapi::mkl::transpose to_blas_(TransposeType trans) {
  switch (trans) {
    case TransposeType::Transpose:
      return oneapi::mkl::transpose::trans;
    case TransposeType::NoTranspose:
      return oneapi::mkl::transpose::nontrans;
    case TransposeType::ConjTranspose:
      return oneapi::mkl::transpose::conjtrans;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

void error_handle(
    int32_t* info_cpu,
    const oneapi::mkl::lapack::batch_error& be) {
  auto errs = be.exceptions();
  auto ids = be.ids();

  for (size_t i = 0; i < errs.size(); ++i) {
    try {
      std::rethrow_exception(errs[i]);
    } catch (const oneapi::mkl::lapack::exception& e) {
      TORCH_WARN(
          "Caught lapack exception:\nWhat: ",
          e.what(),
          "\nInfo: ",
          e.info(),
          "\nDetail: ",
          e.detail());
      info_cpu[ids[i]] = e.info();
    } catch (const sycl::exception& e) {
      TORCH_WARN("Caught SYCL exception:\nWhat: ", e.what(), "\nInfo: -1");
      info_cpu[ids[i]] = -1;
    }
  }
}

template <typename scalar_t>
void mkl_getrf(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    scalar_t* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    int64_t batch_size,
    scalar_t* scratchpad,
    int scratchpadsize) {
  SYCL_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrf_batch,
      queue,
      m,
      n,
      a,
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      batch_size,
      scratchpad,
      scratchpadsize);
}

template <typename scalar_t>
void mkl_getrs(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    scalar_t* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    scalar_t* b,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size,
    scalar_t* scratchpad,
    int64_t scratchpad_size) {
  SYCL_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrs_batch,
      queue,
      trans,
      n,
      nrhs,
      a,
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      b,
      ldb,
      stride_b,
      batch_size,
      scratchpad,
      scratchpad_size);
}

template <typename scalar_t>
int64_t mkl_getrf_scratchpad(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrf_batch_scratchpad_size<scalar_t>(
      queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}

template <typename scalar_t>
int64_t mkl_getrs_scratchpad(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<scalar_t>(
      queue,
      trans,
      n,
      nrhs,
      lda,
      stride_a,
      stride_ipiv,
      ldb,
      stride_b,
      batch_size);
}

template <typename scalar_t>
static void apply_lu_xpu_(
    const Tensor& self_,
    Tensor& pivots_,
    int32_t* info_data) {
  // do nothing if empty input.
  if (self_.numel() == 0)
    return;

  auto& queue = at::xpu::getCurrentSYCLQueue();
  int64_t batch_size = native::batchCount(self_);
  int64_t m = self_.size(-2);
  int64_t n = self_.size(-1);
  int64_t lda = m;
  int64_t stride_a = lda * n;
  int64_t stride_ipiv = (m < n) ? m : n;
  scalar_t* a = reinterpret_cast<scalar_t*>(self_.data_ptr());
  int64_t* ipiv = pivots_.data_ptr<int64_t>();
  int64_t scratchpadsize = mkl_getrf_scratchpad<scalar_t>(
      queue, m, n, lda, stride_a, stride_ipiv, batch_size);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_.options());
  try {
    mkl_getrf<scalar_t>(
        queue,
        m,
        n,
        a,
        lda,
        stride_a,
        ipiv,
        stride_ipiv,
        batch_size,
        reinterpret_cast<scalar_t*>(scratchpad_at.data_ptr()),
        scratchpadsize);
  } catch (const oneapi::mkl::lapack::batch_error& be) {
    error_handle(info_data, be);
  }
}

template <typename scalar_t>
static void apply_lu_solve_xpu_(
    const Tensor& lu_,
    const Tensor& pivots_,
    const Tensor& b_,
    TransposeType t) {
  // do nothing if empty input
  if (lu_.numel() == 0)
    return;

  auto& queue = at::xpu::getCurrentSYCLQueue();
  int64_t batch_size = native::batchCount(b_);

  auto trans = to_blas_(t);
  int64_t n = lu_.size(-2);
  int64_t nrhs = b_.size(-1);
  int64_t lda = lu_.size(-2);
  int64_t stride_a = native::matrixStride(lu_);
  int64_t stride_ipiv = pivots_.size(-1);
  int64_t ldb = b_.size(-2);
  int64_t stride_b = native::matrixStride(b_);

  scalar_t* a = reinterpret_cast<scalar_t*>(lu_.data_ptr());
  Tensor pivots = pivots_;
  if (pivots_.scalar_type() == at::ScalarType::Int)
    pivots = pivots_.to(kLong);
  int64_t* ipiv = pivots.data_ptr<int64_t>();
  scalar_t* b = reinterpret_cast<scalar_t*>(b_.data_ptr());

  std::vector<int32_t> info_vec(batch_size, 0);
  int32_t* info_data = info_vec.data();

  auto execute_mkl_getrs =
      [&](scalar_t* a, scalar_t* b, int64_t* ipiv, int64_t batch_size) {
        int64_t scratchpad_size = mkl_getrs_scratchpad<scalar_t>(
            queue,
            trans,
            n,
            nrhs,
            lda,
            stride_a,
            stride_ipiv,
            ldb,
            stride_b,
            batch_size);
        Tensor scratchpad_at = at::empty({scratchpad_size}, b_.options());
        try {
          mkl_getrs<scalar_t>(
              queue,
              trans,
              n,
              nrhs,
              a,
              lda,
              stride_a,
              ipiv,
              stride_ipiv,
              b,
              ldb,
              stride_b,
              batch_size,
              reinterpret_cast<scalar_t*>(scratchpad_at.data_ptr()),
              scratchpad_size);
        } catch (const oneapi::mkl::lapack::batch_error& be) {
          error_handle(info_data, be);
        }
      };

  bool is_broadcast = false;
  IntArrayRef lu_batch_shape(lu_.sizes().data(), lu_.dim() - 2);
  IntArrayRef b_batch_shape(b_.sizes().data(), b_.dim() - 2);

  {
    auto infer_size_buffer = at::infer_size(lu_batch_shape, b_batch_shape);
    IntArrayRef out_batch_shape(infer_size_buffer);

    is_broadcast = !(out_batch_shape.equals(lu_batch_shape));
  }

  if (!is_broadcast) {
    execute_mkl_getrs(a, b, ipiv, batch_size);
    return;
  }

  BroadcastLinearIndices lu_index(
      native::batchCount(lu_), lu_batch_shape, b_batch_shape);

  for (const auto i : c10::irange(batch_size)) {
    int64_t lu_index_i = lu_index(i);
    scalar_t* a_working_ptr = &a[lu_index_i * stride_a];
    scalar_t* b_working_ptr = &b[i * stride_b];
    int64_t* ipiv_working_ptr = &ipiv[lu_index_i * stride_ipiv];

    execute_mkl_getrs(a_working_ptr, b_working_ptr, ipiv_working_ptr, 1);
  }
}

void lu_solve_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_xpu", [&] {
    using T = get_mkl_type<scalar_t>::type;
    apply_lu_solve_xpu_<T>(LU, pivots, B, trans);
  });
}

void lu_factor_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& info,
    bool pivot) {
  TORCH_CHECK(
      LU.dim() >= 2,
      "torch.lu_factor: Expected tensor with 2 or more dimensions. Got size: ",
      LU.sizes(),
      " instead");
  TORCH_CHECK(
      pivot,
      "linalg.lu_factor: LU without pivoting is not implemented on the XPU");

  // handle the info
  Tensor info_ = at::zeros_like(info, Device(at::kCPU));
  int32_t* info_data = info_.data_ptr<int32_t>();

  // oneMKL requires Long for pivots but PyTorch provides Int
  Tensor pivots_ = at::empty(pivots.sizes(), pivots.options().dtype(kLong));

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_xpu", [&] {
    using T = get_mkl_type<scalar_t>::type;
    apply_lu_xpu_<T>(LU, pivots_, info_data);
  });

  // Copy to original info and pivots tensor
  info.copy_(info_);
  pivots.copy_(pivots_);
}

} // namespace at::native::xpu
