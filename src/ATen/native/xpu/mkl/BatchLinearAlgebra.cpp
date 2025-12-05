#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>
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

  if (!errs.size()) {
    TORCH_WARN(
        "Caught lapack exception:\nWhat: ", be.what(), "\nInfo: ", be.info());
    for (auto& i : ids) {
      TORCH_WARN("Error in matrix #", i);
      info_cpu[i] = 1;
    }
    return;
  }

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
      info_cpu[i] = e.info();
    } catch (const sycl::exception& e) {
      TORCH_WARN("Caught SYCL exception:\nWhat: ", e.what(), "\nInfo: -1");
      info_cpu[i] = -1;
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

template <>
void mkl_getrf<c10::complex<double>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    c10::complex<double>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    int64_t batch_size,
    c10::complex<double>* scratchpad,
    int scratchpadsize) {
  SYCL_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrf_batch,
      queue,
      m,
      n,
      reinterpret_cast<std::complex<double>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      batch_size,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpadsize);
}

template <>
void mkl_getrf<c10::complex<float>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    c10::complex<float>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    int64_t batch_size,
    c10::complex<float>* scratchpad,
    int scratchpadsize) {
  SYCL_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrf_batch,
      queue,
      m,
      n,
      reinterpret_cast<std::complex<float>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      batch_size,
      reinterpret_cast<std::complex<float>*>(scratchpad),
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

template <>
void mkl_getrs<c10::complex<double>>(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    c10::complex<double>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    c10::complex<double>* b,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size,
    c10::complex<double>* scratchpad,
    int64_t scratchpad_size) {
  SYCL_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrs_batch,
      queue,
      trans,
      n,
      nrhs,
      reinterpret_cast<std::complex<double>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      reinterpret_cast<std::complex<double>*>(b),
      ldb,
      stride_b,
      batch_size,
      reinterpret_cast<std::complex<double>*>(scratchpad),
      scratchpad_size);
}

template <>
void mkl_getrs<c10::complex<float>>(
    sycl::queue& queue,
    oneapi::mkl::transpose trans,
    int64_t n,
    int64_t nrhs,
    c10::complex<float>* a,
    int64_t lda,
    int64_t stride_a,
    int64_t* ipiv,
    int64_t stride_ipiv,
    c10::complex<float>* b,
    int64_t ldb,
    int64_t stride_b,
    int64_t batch_size,
    c10::complex<float>* scratchpad,
    int64_t scratchpad_size) {
  SYCL_ONEMKL_SUBMIT(
      queue,
      oneapi::mkl::lapack::getrs_batch,
      queue,
      trans,
      n,
      nrhs,
      reinterpret_cast<std::complex<float>*>(a),
      lda,
      stride_a,
      ipiv,
      stride_ipiv,
      reinterpret_cast<std::complex<float>*>(b),
      ldb,
      stride_b,
      batch_size,
      reinterpret_cast<std::complex<float>*>(scratchpad),
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

template <>
int64_t mkl_getrf_scratchpad<c10::complex<double>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrf_batch_scratchpad_size<std::complex<double>>(
      queue, m, n, lda, stride_a, stride_ipiv, batch_size);
}

template <>
int64_t mkl_getrf_scratchpad<c10::complex<float>>(
    sycl::queue& queue,
    int64_t m,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getrf_batch_scratchpad_size<std::complex<float>>(
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

template <>
int64_t mkl_getrs_scratchpad<c10::complex<double>>(
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
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<double>>(
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

template <>
int64_t mkl_getrs_scratchpad<c10::complex<float>>(
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
  return oneapi::mkl::lapack::getrs_batch_scratchpad_size<std::complex<float>>(
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
  scalar_t* a = (scalar_t*)(self_.data_ptr());
  int64_t* ipiv = (int64_t*)(pivots_.data_ptr());
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
        (scalar_t*)(scratchpad_at.data_ptr()),
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

  scalar_t* a = lu_.data_ptr<scalar_t>();
  Tensor pivots = pivots_;
  if (pivots_.scalar_type() == at::ScalarType::Int)
    pivots = pivots_.to(kLong);
  int64_t* ipiv = pivots.data_ptr<int64_t>();
  scalar_t* b = b_.data_ptr<scalar_t>();

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
              scratchpad_at.data_ptr<scalar_t>(),
              scratchpad_size);
        } catch (oneapi::mkl::lapack::batch_error be) {
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
    apply_lu_solve_xpu_<scalar_t>(LU, pivots, B, trans);
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
    apply_lu_xpu_<scalar_t>(LU, pivots_, info_data);
  });

  // Copy to original info and pivots tensor
  info.copy_(info_);
  pivots.copy_(pivots_);
}


template <typename scalar_t>
void linalg_qr_kernel_impl(
    const at::Tensor& A,
    std::string_view mode,
    const at::Tensor& Q,
    const at::Tensor& R) {


  at::Tensor a_contig = A.contiguous();
  at::Tensor result_r = at::clone(a_contig);

  auto options = at::TensorOptions().dtype(A.dtype()).device(kXPU);
  auto dimensions = A.sizes();

  result_r = result_r.transpose(-2, -1).contiguous();

  int numel = a_contig.numel();
  int range = a_contig.dim();
  int64_t n = a_contig.sizes().at(range - 2);
  int64_t m = a_contig.sizes().at(range - 1);
  int64_t mn = int64_t(m * n);
  int64_t b = numel ==0 ? 0 : numel / mn;


  if (b==0 && mode=="complete" && n>0) {
    b=1;
    for (int dimension=0; dimension<range-2; dimension++) {
      b*=a_contig.sizes().at(dimension);
    }
  }

  int out_q_columns = m > n ? n : m;
  if (n > m && mode == "complete") {
    out_q_columns = n;
  }

  std::vector v(dimensions.begin(), dimensions.end());
  if (mode != "r") {
    v[range - 1] = v[range - 2];
    v[range - 2] = out_q_columns;
  } else {
    v = std::vector<long>({0});
  }
  auto q_dimensions = at::IntArrayRef(v);

  at::Tensor result_q = at::empty(q_dimensions, options);



  sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

  int64_t bufsize1 =
      oneapi::mkl::lapack::geqrf_scratchpad_size<scalar_t>(queue, n+1, m+1, n+1);
  int64_t bufsize2 =
      oneapi::mkl::lapack::orgqr_scratchpad_size<scalar_t>(queue, n+1, m+1, m+1, n+1);

  int64_t bufsize = bufsize2 > bufsize1 ? bufsize2 : bufsize1;
  int64_t tau_len = m > n ? n : m;
  scalar_t* sbuffer = sycl::malloc_device<scalar_t>(bufsize, queue);
  scalar_t* tau_buf = sycl::malloc_device<scalar_t>(tau_len, queue);
  scalar_t* r_buf = result_r.data_ptr<scalar_t>();

  scalar_t* q_buf = nullptr;
  if (mode != "r") {
    q_buf = result_q.data_ptr<scalar_t>();
  }

  for (int batch_item = 0; batch_item < b; batch_item++) {


    if (mn!=0) // make QR if there is something to orthogonalize
      oneapi::mkl::lapack::geqrf(queue, n, m, r_buf, n, tau_buf, sbuffer, bufsize)
        .wait();

    if (mode != "r") {
      // copy relevant part of R matrix to Q matrix
      int copy_columns = out_q_columns > m ? m : out_q_columns;
      queue.memcpy(q_buf, r_buf, n * copy_columns * sizeof(scalar_t)).wait();

      oneapi::mkl::lapack::orgqr(
          queue,
          n,
          out_q_columns,
          tau_len,
          q_buf,
          n,
          tau_buf,
          sbuffer,
          bufsize)
          .wait();

      q_buf += n * out_q_columns;
    }

    r_buf += mn;

  } // batch

  sycl::free(sbuffer, queue);
  sycl::free(tau_buf, queue);

  if ((mode == "reduced" || mode == "r") && n > m) {
    result_r =
        result_r
            .index(
                {"...", at::indexing::Slice(0, n), at::indexing::Slice(0, m)})
            .contiguous();
  }

  // normal case, non-zero dimensions
  if (mode!="r") {
    result_q.transpose_(-2, -1);
  }
  Q.set_(result_q);
  R.set_(result_r.transpose(-2, -1).triu_());
  queue.wait();
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
  //
