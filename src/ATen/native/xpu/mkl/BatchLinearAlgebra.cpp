/*
 * Copyright 2020-2026 Intel Corporation
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
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/isnan.h>
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

// Create NaN value that works for both real and complex types
template <typename scalar_t>
inline scalar_t create_quiet_nan() {
  using real_t = typename c10::scalar_value_type<scalar_t>::type;
  real_t nan_val = std::numeric_limits<real_t>::quiet_NaN();
  if constexpr (c10::is_complex<scalar_t>::value) {
    return scalar_t(nan_val, nan_val);
  } else {
    return nan_val;
  }
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
    if (!at::isnan(LU).any().item<bool>()) {
      apply_lu_xpu_<T>(LU, pivots_, info_data);
    } else {
      // Has NaN, temporarily replace NaNs to avoid MKL crashes, run batched LU
      // then restore NaNs for the affected batches.
      int64_t batch_size = native::batchCount(LU);
      int64_t m = LU.size(-2);
      int64_t n = LU.size(-1);

      // Detect NaN per-batch
      auto nan_mask_batch =
          at::isnan(LU).reshape({batch_size, m * n}).any(/*dim=*/1);

      // Replace NaN batches with identity matrix to avoid MKL crash
      auto identity = at::eye(m, n, LU.options()).unsqueeze(0);
      auto nan_mask_expanded = nan_mask_batch.view({batch_size, 1, 1});
      LU.copy_(at::where(nan_mask_expanded, identity, LU));

      apply_lu_xpu_<T>(LU, pivots_, info_data);

      // Restore NaN for batches that originally had NaN
      LU.masked_fill_(
          nan_mask_expanded.expand({batch_size, m, n}),
          create_quiet_nan<scalar_t>());
    }
  });

  // Copy to original info and pivots tensor
  info.copy_(info_);
  pivots.copy_(pivots_);
}

template <typename T>
void apply_triangular_solve_mkl(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  auto& queue = at::xpu::getCurrentSYCLQueue();

  oneapi::mkl::side left_right =
      left ? oneapi::mkl::side::left : oneapi::mkl::side::right;
  oneapi::mkl::uplo upper_lower =
      upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
  oneapi::mkl::transpose transa = to_blas_(transpose);
  oneapi::mkl::diag unit_diag =
      unitriangular ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;

  const int64_t batch_size = batchCount(A);
  const int64_t m = left ? A.size(-1) : B.size(-2);
  const int64_t n = B.size(-1);
  const int64_t lda = std::max<int64_t>(1, A.size(-2));
  const int64_t ldb = std::max<int64_t>(1, B.size(-2));

  const T* A_data = reinterpret_cast<const T*>(A.const_data_ptr());
  T* B_data = reinterpret_cast<T*>(B.data_ptr());

  if (batch_size > 1) {
    const int64_t A_mat_stride = matrixStride(A);
    const int64_t B_mat_stride = matrixStride(B);

    oneapi::mkl::blas::column_major::trsm_batch(
        queue,
        left_right,
        upper_lower,
        transa,
        unit_diag,
        m,
        n,
        T(1),
        A_data,
        lda,
        A_mat_stride,
        B_data,
        ldb,
        B_mat_stride,
        batch_size);
  } else {
    oneapi::mkl::blas::column_major::trsm(
        queue,
        left_right,
        upper_lower,
        transa,
        unit_diag,
        m,
        n,
        T(1),
        A_data,
        lda,
        B_data,
        ldb);
  }
}

void triangular_solve_mkl(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  if (A.numel() == 0 || B.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      A.scalar_type(), "triangular_solve_mkl", [&] {
        using T = get_mkl_type<scalar_t>::type;
        apply_triangular_solve_mkl<T>(
            A, B, left, upper, transpose, unitriangular);
      });
}

template <typename scalar_t>
void linalg_qr_kernel_impl(
    const at::Tensor& A,
    std::string_view mode,
    const at::Tensor& Q,
    const at::Tensor& R) {
  at::Tensor a_contig = A.contiguous();

  auto options = at::TensorOptions().dtype(A.dtype()).device(kXPU);
  auto dimensions = A.sizes();

  int64_t numel = a_contig.numel();
  int64_t range = a_contig.dim();
  int64_t n = a_contig.size(-2);
  int64_t m = a_contig.size(-1);
  int64_t mn = m * n;
  int64_t b = numel == 0 ? 0 : numel / mn;

  // Prepare R output matrix - correct dimensions if needed
  at::Tensor r_out_;

  if (numel == 0 && mode != "complete") {
    std::vector r_sizes(dimensions.begin(), dimensions.end());
    if (r_sizes[range - 1] == 0) {
      r_sizes[range - 2] = 0;
    }
    r_out_ = at::zeros(r_sizes, options);
  } else {
    r_out_ = at::clone(a_contig);
  }

  r_out_ = r_out_.transpose(-2, -1).contiguous();

  int64_t out_q_columns = std::min(m, n);
  if (n > m && mode == "complete") {
    out_q_columns = n;
  }

  // Prepare Q output matrix - correct dimensions if needed
  std::vector q_sizes(dimensions.begin(), dimensions.end());
  if (mode != "r") {
    q_sizes[range - 1] = q_sizes[range - 2];
    q_sizes[range - 2] = out_q_columns;
  } else {
    // dim = (0) for "r" mode
    q_sizes = std::vector<long>({0});
  }

  at::Tensor q_out_ = at::empty(q_sizes, options);

  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

  // add one to size to avoid special case when any of dimensions is 0.
  int64_t bufsize1 = oneapi::mkl::lapack::geqrf_scratchpad_size<scalar_t>(
      queue, n + 1, m + 1, n + 1);
  int64_t bufsize2 = oneapi::mkl::lapack::orgqr_scratchpad_size<scalar_t>(
      queue, n + 1, m + 1, m + 1, n + 1);

  int64_t bufsize = std::max(bufsize2, bufsize1);
  int64_t tau_len = std::min(m, n);
  scalar_t* sbuffer = sycl::malloc_device<scalar_t>(bufsize, queue);
  scalar_t* tau_buf = sycl::malloc_device<scalar_t>(tau_len, queue);
  scalar_t* r_buf = r_out_.data_ptr<scalar_t>();

  scalar_t* q_buf = nullptr;
  if (mode != "r") {
    q_buf = q_out_.data_ptr<scalar_t>();
  }

  if (b == 0 && mode == "complete" && n > 0) {
    b = native::batchCount(a_contig);
  }

  for (int batch_item = 0; batch_item < b; batch_item++) {
    if (mn != 0) { // make QR if there is something to orthogonalize
      oneapi::mkl::lapack::geqrf(
          queue, n, m, r_buf, n, tau_buf, sbuffer, bufsize)
          .wait();
    }

    if (mode != "r") {
      // copy relevant part of R matrix to Q matrix
      int64_t copy_columns = std::min(out_q_columns, m);
      queue.memcpy(q_buf, r_buf, n * copy_columns * sizeof(scalar_t)).wait();

      oneapi::mkl::lapack::orgqr(
          queue, n, out_q_columns, tau_len, q_buf, n, tau_buf, sbuffer, bufsize)
          .wait();

      q_buf += n * out_q_columns;
    }

    r_buf += mn;
  } // batch

  sycl::free(sbuffer, queue);
  sycl::free(tau_buf, queue);

  if ((mode == "reduced" || mode == "r") && n > m) {
    r_out_ =
        r_out_
            .index(
                {"...", at::indexing::Slice(0, n), at::indexing::Slice(0, m)})
            .contiguous();
  }

  // normal case, non-zero dimensions
  if (mode != "r") {
    q_out_ = q_out_.transpose_(-2, -1).contiguous();
  }

  Q.set_(q_out_);
  R.set_(r_out_.transpose(-2, -1).triu_());
}

void linalg_qr_kernel(
    const at::Tensor& A,
    std::string_view mode,
    const at::Tensor& Q,
    const at::Tensor& R) {
  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "linalg_qr_xpu", [&] {
    linalg_qr_kernel_impl<scalar_t>(A, mode, Q, R);
  });
}

} // namespace at::native::xpu
