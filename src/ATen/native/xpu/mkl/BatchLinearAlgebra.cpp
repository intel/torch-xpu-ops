#if defined(USE_ONEMKL)
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
#include <ATen/ops/_linalg_svd.h>
#include <ATen/ops/_linalg_svd_meta.h>
#include <ATen/ops/_linalg_svd_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_lu_factor_ex_meta.h>
#include <ATen/ops/linalg_lu_factor_ex_native.h>
#include <ATen/ops/linalg_lu_factor_native.h>
#include <ATen/ops/linalg_lu_meta.h>
#include <ATen/ops/linalg_lu_native.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_lu_solve_meta.h>
#include <ATen/ops/linalg_lu_solve_native.h>
#include <ATen/ops/lu_solve_native.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/where.h>
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

static inline std::tuple<Tensor, Tensor, Tensor> _create_U_S_VT(
    const Tensor& input,
    bool some,
    bool compute_uv) {
  auto sizes = input.sizes().vec();
  int64_t m = input.size(-2), n = input.size(-1);
  auto k = std::min(m, n);

  sizes[input.dim() - 1] = (compute_uv && some) ? k : m;
  auto U_strides =
      at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
  // U should be a column-major or a batch of column-major matrices
  // ... x m x ucol will have strides: ...., ucol, 1
  // We require: ...., 1, m

  Tensor U_empty;
  U_empty = at::empty_strided(sizes, U_strides, input.options());

  sizes[input.dim() - 2] = some ? k : n;
  sizes[input.dim() - 1] = n;
  auto Vh_strides =
      at::native::batched_matrix_contiguous_strides(sizes, /*f-contig*=*/true);
  Tensor VT_empty;
  VT_empty = at::empty_strided(sizes, Vh_strides, input.options());

  sizes.pop_back();
  sizes[input.dim() - 2] = std::min(m, n);
  Tensor S_empty;
  ScalarType dtype = toRealValueType(typeMetaToScalarType(input.dtype()));
  S_empty = at::empty(sizes, input.options().dtype(dtype));
  return std::tuple<Tensor, Tensor, Tensor>(U_empty, S_empty, VT_empty);
}

template <typename scalar_t, typename value_t>
static void apply_svd(
    sycl::queue& queue,
    scalar_t* self_data,
    int64_t lda,
    int64_t self_stride,
    int64_t batchsize,
    int64_t m,
    int64_t n,
    TensorOptions self_opt,
    scalar_t* U_data,
    int64_t ldu,
    int64_t U_stride,
    value_t* S_data,
    int64_t S_stride,
    scalar_t* VT_data,
    int64_t ldvt,
    int64_t VT_stride,
    char jobz) {
  oneapi::mkl::jobsvd jobu, jobvt;
  if (jobz == 'N') {
    jobu = oneapi::mkl::jobsvd::N;
    jobvt = oneapi::mkl::jobsvd::N;
  } else if (jobz == 'S') {
    jobu = oneapi::mkl::jobsvd::S;
    jobvt = oneapi::mkl::jobsvd::S;
  } else {
    jobu = oneapi::mkl::jobsvd::A;
    jobvt = oneapi::mkl::jobsvd::A;
  }

  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::gesvd_scratchpad_size<scalar_t>(
          queue, jobu, jobvt, m, n, lda, ldu, ldvt);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_opt);

  for (int64_t i = 0; i < batchsize; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_stride];
    scalar_t* U_working_ptr = &U_data[i * U_stride];
    value_t* S_working_ptr = &S_data[i * S_stride];
    scalar_t* VT_working_ptr = &VT_data[i * VT_stride];

    SYCL_ONEMKL_SUBMIT(
        queue,
        oneapi::mkl::lapack::gesvd,
        queue,
        jobu,
        jobvt,
        m,
        n,
        self_working_ptr,
        lda,
        S_working_ptr,
        U_working_ptr,
        ldu,
        VT_working_ptr,
        ldvt,
        (scalar_t*)(scratchpad_at.data_ptr()),
        scratchpadsize);
  }
} // namespace impl

template <>
void apply_svd<c10::complex<double>, double>(
    sycl::queue& queue,
    c10::complex<double>* self_data,
    int64_t lda,
    int64_t self_stride,
    int64_t batchsize,
    int64_t m,
    int64_t n,
    TensorOptions self_opt,
    c10::complex<double>* U_data,
    int64_t ldu,
    int64_t U_stride,
    double* S_data,
    int64_t S_stride,
    c10::complex<double>* VT_data,
    int64_t ldvt,
    int64_t VT_stride,
    char jobz) {
  oneapi::mkl::jobsvd jobu, jobvt;
  if (jobz == 'N') {
    jobu = oneapi::mkl::jobsvd::N;
    jobvt = oneapi::mkl::jobsvd::N;
  } else if (jobz == 'S') {
    jobu = oneapi::mkl::jobsvd::S;
    jobvt = oneapi::mkl::jobsvd::S;
  } else {
    jobu = oneapi::mkl::jobsvd::A;
    jobvt = oneapi::mkl::jobsvd::A;
  }

  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<double>>(
          queue, jobu, jobvt, m, n, lda, ldu, ldvt);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_opt);

  for (int64_t i = 0; i < batchsize; i++) {
    c10::complex<double>* self_working_ptr = &self_data[i * self_stride];
    c10::complex<double>* U_working_ptr = &U_data[i * U_stride];
    double* S_working_ptr = &S_data[i * S_stride];
    c10::complex<double>* VT_working_ptr = &VT_data[i * VT_stride];

    SYCL_ONEMKL_SUBMIT(
        queue,
        oneapi::mkl::lapack::gesvd,
        queue,
        jobu,
        jobvt,
        m,
        n,
        reinterpret_cast<std::complex<double>*>(self_working_ptr),
        lda,
        S_working_ptr,
        reinterpret_cast<std::complex<double>*>(U_working_ptr),
        ldu,
        reinterpret_cast<std::complex<double>*>(VT_working_ptr),
        ldvt,
        reinterpret_cast<std::complex<double>*>(scratchpad_at.data_ptr()),
        scratchpadsize);
  }
}

template <>
void apply_svd<c10::complex<float>, float>(
    sycl::queue& queue,
    c10::complex<float>* self_data,
    int64_t lda,
    int64_t self_stride,
    int64_t batchsize,
    int64_t m,
    int64_t n,
    TensorOptions self_opt,
    c10::complex<float>* U_data,
    int64_t ldu,
    int64_t U_stride,
    float* S_data,
    int64_t S_stride,
    c10::complex<float>* VT_data,
    int64_t ldvt,
    int64_t VT_stride,
    char jobz) {
  oneapi::mkl::jobsvd jobu, jobvt;
  if (jobz == 'N') {
    jobu = oneapi::mkl::jobsvd::N;
    jobvt = oneapi::mkl::jobsvd::N;
  } else if (jobz == 'S') {
    jobu = oneapi::mkl::jobsvd::S;
    jobvt = oneapi::mkl::jobsvd::S;
  } else {
    jobu = oneapi::mkl::jobsvd::A;
    jobvt = oneapi::mkl::jobsvd::A;
  }

  std::int64_t scratchpadsize =
      oneapi::mkl::lapack::gesvd_scratchpad_size<std::complex<float>>(
          queue, jobu, jobvt, m, n, lda, ldu, ldvt);
  Tensor scratchpad_at = at::empty({scratchpadsize}, self_opt);

  for (int64_t i = 0; i < batchsize; i++) {
    c10::complex<float>* self_working_ptr = &self_data[i * self_stride];
    c10::complex<float>* U_working_ptr = &U_data[i * U_stride];
    float* S_working_ptr = &S_data[i * S_stride];
    c10::complex<float>* VT_working_ptr = &VT_data[i * VT_stride];

    SYCL_ONEMKL_SUBMIT(
        queue,
        oneapi::mkl::lapack::gesvd,
        queue,
        jobu,
        jobvt,
        m,
        n,
        reinterpret_cast<std::complex<float>*>(self_working_ptr),
        lda,
        S_working_ptr,
        reinterpret_cast<std::complex<float>*>(U_working_ptr),
        ldu,
        reinterpret_cast<std::complex<float>*>(VT_working_ptr),
        ldvt,
        reinterpret_cast<std::complex<float>*>(scratchpad_at.data_ptr()),
        scratchpadsize);
  }
}

std::tuple<Tensor, Tensor, Tensor> _svd_helper(
    const Tensor& self,
    bool some,
    bool compute_uv) {
  auto infos_tensor = at::zeros(
      native::batchCount(self),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);

  char jobz = compute_uv ? (some ? 'S' : 'A') : 'N';

  Tensor U_working_copy, S_working_copy, VT_working_copy;
  std::tie(U_working_copy, S_working_copy, VT_working_copy) =
      _create_U_S_VT(self, some, compute_uv);

  if (self.numel() > 0) {
    auto self_working_copy = native::cloneBatchedColumnMajor(self);
    auto& queue = at::xpu::getCurrentSYCLQueue();
    auto self_stride = at::native::matrixStride(self_working_copy);
    auto U_stride = compute_uv ? at::native::matrixStride(U_working_copy) : 1;
    auto S_stride = S_working_copy.size(-1);
    auto VT_stride = compute_uv ? at::native::matrixStride(VT_working_copy) : 1;
    auto batchsize = at::native::batchCount(self_working_copy);

    auto m = self_working_copy.size(-2);
    auto n = self_working_copy.size(-1);
    int64_t lda = self_working_copy.stride(-1);
    int64_t ldu = compute_uv ? U_working_copy.stride(-1) : 1;
    int64_t ldvt = compute_uv ? VT_working_copy.stride(-1) : 1;
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "svd_xpu", [&] {
      using value_t = typename c10::scalar_value_type<scalar_t>::type;
      apply_svd<scalar_t, value_t>(
          queue,
          self_working_copy.data_ptr<scalar_t>(),
          lda,
          self_stride,
          batchsize,
          m,
          n,
          self.options(),
          compute_uv ? U_working_copy.data_ptr<scalar_t>() : nullptr,
          ldu,
          U_stride,
          S_working_copy.data_ptr<value_t>(),
          S_stride,
          compute_uv ? VT_working_copy.data_ptr<scalar_t>() : nullptr,
          ldvt,
          VT_stride,
          jobz);
    });

    std::copy(
        infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
    at::_linalg_check_errors(infos_tensor, "svd_xpu", self.dim() == 2);

    if (!compute_uv) {
      VT_working_copy.zero_();
      U_working_copy.zero_();
    }
  } else {
    U_working_copy.zero_();
    VT_working_copy.zero_();
  }

  return std::make_tuple(U_working_copy, S_working_copy, VT_working_copy);
}

static void svd_resize_and_copy(
    const char* name,
    const Tensor& src,
    const Tensor& dst) {
  TORCH_CHECK(
      src.device() == dst.device(),
      "svd output tensor ",
      name,
      " is on the wrong device: expected ",
      src.device(),
      " got ",
      dst.device());
  at::native::resize_output(dst, src.sizes());
  dst.copy_(src);
}

void svd_mkl(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const c10::optional<c10::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const Tensor& info) {
  Tensor U_tmp, S_tmp, Vh_tmp;
  bool some = !full_matrices;
  std::tie(U_tmp, S_tmp, Vh_tmp) = _svd_helper(A, some, compute_uv);

  // TODO: Remove copy
  if (compute_uv) {
    svd_resize_and_copy("U", U_tmp, U);
    svd_resize_and_copy("Vh", Vh_tmp, Vh);
  }
  svd_resize_and_copy("S", S_tmp, S);
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
    std::vector<int32_t>& infos,
    oneapi::mkl::lapack::batch_error& be) {
  auto errs = be.exceptions();
  auto ids = be.ids();
  for (auto& i : ids) {
    try {
      std::rethrow_exception(errs[i]);
    } catch (oneapi::mkl::lapack::exception e) {
      std::cout << "Cathed lapack exception:"
                << "\nWhat: " << e.what() << "\nInfo: " << e.info()
                << "\nDetail: " << e.detail() << std::endl;
      infos[i] = e.info();
    } catch (sycl::exception e) {
      std::cout << "Catched SYCL exception:"
                << "\nWhat: " << e.what() << "\nInfo: -1" << std::endl;
      infos[i] = -1;
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
int64_t mkl_getri_scratchpad(
    sycl::queue& queue,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<scalar_t>(
      queue, n, lda, stride_a, stride_ipiv, ldainv, stride_ainv, batch_size);
}

template <>
int64_t mkl_getri_scratchpad<c10::complex<double>>(
    sycl::queue& queue,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<double>>(
      queue, n, lda, stride_a, stride_ipiv, ldainv, stride_ainv, batch_size);
}

template <>
int64_t mkl_getri_scratchpad<c10::complex<float>>(
    sycl::queue& queue,
    int64_t n,
    int64_t lda,
    int64_t stride_a,
    int64_t stride_ipiv,
    int64_t ldainv,
    int64_t stride_ainv,
    int64_t batch_size) {
  return oneapi::mkl::lapack::getri_batch_scratchpad_size<std::complex<float>>(
      queue, n, lda, stride_a, stride_ipiv, ldainv, stride_ainv, batch_size);
}

template <typename scalar_t>
static void apply_lu_xpu_(
    Tensor& self_,
    Tensor& pivots_,
    std::vector<int32_t>& infos_) {
#ifdef USE_ONEMKL
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
  } catch (oneapi::mkl::lapack::batch_error be) {
    error_handle(infos_, be);
  }
#else
  AT_ERROR("lu: oneMKL library not found in compilation");
#endif
}

template <typename scalar_t>
static void apply_lu_solve_xpu_(
    const Tensor& b_,
    const Tensor& lu_,
    const Tensor& pivots_,
    std::vector<int32_t>& infos_,
    TransposeType t) {
#ifdef USE_ONEMKL
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

  scalar_t* a = (scalar_t*)(lu_.data_ptr());
  Tensor pivots = pivots_;
  if (pivots_.scalar_type() == at::ScalarType::Int)
    pivots = pivots_.to(kLong);
  int64_t* ipiv = (int64_t*)(pivots.data_ptr());
  scalar_t* b = (scalar_t*)(b_.data_ptr());

  int64_t scratchpadsize = mkl_getrs_scratchpad<scalar_t>(
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
  Tensor scratchpad_at = at::empty({scratchpadsize}, b_.options());
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
        (scalar_t*)(scratchpad_at.data_ptr()),
        scratchpadsize);
  } catch (oneapi::mkl::lapack::batch_error be) {
    error_handle(infos_, be);
  }
#else
  AT_ERROR("lu: oneMKL library not found in compilation");
#endif
}

std::tuple<Tensor, Tensor, Tensor> _lu_with_info(
    const Tensor& self,
    bool pivot,
    bool check_errors) {
  TORCH_CHECK(
      self.dim() >= 2,
      "expected tensor with 2 or more dimensions, got size: ",
      self.sizes(),
      " instead");
  auto m = self.size(-2);
  auto n = self.size(-1);
  auto req_size = self.sizes().vec();
  req_size.pop_back();
  req_size.back() = std::min(m, n);
  auto pivots_tensor = at::empty(req_size, self.options().dtype(kLong));
  req_size.pop_back();
  auto infos_tensor =
      at::zeros(req_size, self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);

  Tensor self_working_copy;
  if (self.numel() == 0) {
    self_working_copy = at::empty_like(self);
  } else {
    self_working_copy = native::cloneBatchedColumnMajor(self);
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(self.scalar_type(), "lu_xpu", [&] {
      apply_lu_xpu_<scalar_t>(self_working_copy, pivots_tensor, infos);
    });
  }
  if (check_errors) {
    at::_linalg_check_errors(infos_tensor, "lu_xpu", self.dim() == 2);
  }
  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  return std::make_tuple(
      self_working_copy, pivots_tensor.to(kInt), infos_tensor);
}

// Solves a system of linear equations matmul(input, x) = other in-place
static Tensor& linalg_solve_out_info(
    Tensor& result,
    Tensor& infos,
    const Tensor& input,
    const Tensor& other) {
  at::native::checkSameDevice("linalg_solve", result, input);
  at::native::checkSameDevice("linalg_solve", other, input, "other");
  at::native::checkLinalgCompatibleDtype("linalg_solve", result, input);

  TORCH_CHECK(
      input.scalar_type() == other.scalar_type(),
      "input dtype ",
      input.scalar_type(),
      " does not match other dtype ",
      other.scalar_type());

  TORCH_CHECK(
      input.dim() >= 2,
      "input should have at least 2 dimensions, but has ",
      input.dim(),
      " dimensions instead");
  TORCH_CHECK(
      other.dim() >= 1,
      "other should have at least 1 dimension, but has ",
      other.dim(),
      " dimensions instead");

  // Two types of 'other' tensors are supported:
  // - 1-dimensional (1D) tensor or batch of 1D tensors (vector case)
  // - 2-dimensional (2D) tensor or batch of 2D tensors (matrix case)
  // original torch.solve supported only the matrix case, while NumPy works for
  // both cases for the batched input we need to be able to distinguish them
  bool vector_case = at::native::linalg_solve_is_vector_rhs(input, other);

  bool is_batched_column_major = false;
  if (vector_case) {
    is_batched_column_major = result.is_contiguous();
  } else if (!vector_case && result.dim() >= 2) {
    is_batched_column_major = result.transpose(-2, -1).is_contiguous();
  }

  // if 'other' is a batch of 2D tensors, then 'input' can be non-batched and
  // will be broadcasted
  auto expected_shape =
      IntArrayRef(input.sizes().data(), input.dim() - 1); // input.shape[:-1]
  if (!vector_case && other.dim() > 2) {
    expected_shape = other.sizes();
  }

  bool result_equal_expected_shape = result.sizes().equals(expected_shape);
  bool result_input_same_type = (result.scalar_type() == input.scalar_type());

  // if result is not empty and not in batched column major format
  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= !result_input_same_type; // or result does not have the same
                                          // dtype as input
  copy_needed |=
      (result.numel() != 0 &&
       !result_equal_expected_shape); // or result does not have the expected
                                      // shape
  // we have to allocate a temporary tensor
  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, input.options());
    result_tmp = linalg_solve_out_info(result_tmp, infos, input, other);
    resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
    return result;
  }
  // else use result's storage directly

  // we need to unsqueeze 'other' because 2-dimensional tensors are expected in
  // the implementation
  Tensor other_ = vector_case ? other.unsqueeze(-1) : other;

  // _linalg_broadcast_batch_dims also includes linearSolveCheckInputs
  // it checks for squareness of 'input' and 'shape' compatibility of 'other'
  // and 'input'
  Tensor other_broadcasted, input_broadcasted;
  std::tie(other_broadcasted, input_broadcasted) =
      at::native::_linalg_broadcast_batch_dims(other_, input, "linalg_solve");

  auto squeezed_other_broadcasted = at::squeeze(other_broadcasted, -1);
  auto squeezed_result_shape = squeezed_other_broadcasted.sizes();

  // if result has no elements we can modify it
  if (result.numel() == 0) {
    if (vector_case) {
      result.resize_(squeezed_result_shape);
    } else {
      at::native::resize_as_(
          result,
          other_broadcasted.transpose(-2, -1),
          MemoryFormat::Contiguous);
      result.transpose_(-2, -1);
    }
  }

  auto expected_result_shape =
      vector_case ? squeezed_result_shape : other_broadcasted.sizes();
  TORCH_INTERNAL_ASSERT(result.sizes().equals(expected_result_shape));
  TORCH_INTERNAL_ASSERT(result.scalar_type() == input.scalar_type());
  TORCH_INTERNAL_ASSERT(result.device() == input.device());

  // result tensor must be in batched column major order (Fortran contiguous)
  // for 2D inputs or C contiguous for 1D input
  if (vector_case) {
    TORCH_INTERNAL_ASSERT(result.is_contiguous());
  } else {
    TORCH_INTERNAL_ASSERT(result.transpose(-2, -1).is_contiguous());
  }

  // for 1-dimensional 'other', we need to unsqueeze the result before passing
  // to "apply_solve"
  if (vector_case) {
    result = result.unsqueeze_(-1);
  }

  // lu_stub+lu_solve_stub perform calculations in-place and 'result' must be a
  // copy of 'other_broadcasted'
  result.copy_(other_broadcasted);

  auto input_working_copy =
      at::native::cloneBatchedColumnMajor(input_broadcasted);

  infos.resize_({std::max<int64_t>(1, native::batchCount(input_broadcasted))})
      .zero_();
  std::vector<int32_t> infos_vec_1(native::batchCount(input_broadcasted), 0);
  std::vector<int32_t> infos_vec_2(native::batchCount(input_broadcasted), 0);
  // compute the LU factorization of 'input_working_copy'
  auto pivots_shape =
      IntArrayRef(input_broadcasted.sizes().data(), input_broadcasted.dim() - 2)
          .vec(); // input_broadcasted.shape[:-2]
  pivots_shape.push_back(std::min(input.size(-2), input.size(-1)));
  Tensor pivots = at::empty(pivots_shape, input.options().dtype(kLong));
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input_working_copy.scalar_type(), "linalg_solve_xpu", [&] {
        apply_lu_xpu_<scalar_t>(input_working_copy, pivots, infos_vec_1);
        // solve the linear system using the LU factorization
        apply_lu_solve_xpu_<scalar_t>(
            result,
            input_working_copy,
            pivots,
            infos_vec_2,
            TransposeType::NoTranspose);
      });

  std::copy(
      infos_vec_1.begin(),
      infos_vec_1.end(),
      infos.template data_ptr<int32_t>());

  at::_linalg_check_errors(
      infos, "lu_solve_xpu", input_working_copy.dim() == 2);

  // for 1-dimensional 'other', we need to squeeze the result after
  // "apply_solve"
  if (vector_case) {
    result = result.squeeze_(-1);
  }

  return result;
}

Tensor _lu_solve_helper(
    const Tensor& self,
    const Tensor& LU_data,
    const Tensor& LU_pivots) {
  auto self_working_copy = native::cloneBatchedColumnMajor(self);
  auto LU_data_working_copy = native::cloneBatchedColumnMajor(LU_data);
  auto LU_pivots_working_copy =
      LU_pivots.is_contiguous() ? LU_pivots : LU_pivots.contiguous();
  // FIXME: oneMKL only support int64_t datatype of pivots
  LU_pivots_working_copy = LU_pivots.to(kLong);
  auto infos_tensor = at::zeros(
      native::batchCount(self),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(self), 0);

  if (self.numel() == 0 || LU_data.numel() == 0) {
    return at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "lu_solve_xpu", [&] {
    apply_lu_solve_xpu_<scalar_t>(
        self_working_copy,
        LU_data_working_copy,
        LU_pivots_working_copy,
        infos,
        TransposeType::NoTranspose);
  });

  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(infos_tensor, "lu_solve_xpu", self.dim() == 2);

  return self_working_copy;
}

Tensor _det_lu_based_helper_backward_helper(
    const Tensor& det_grad,
    const Tensor& det,
    const Tensor& self,
    const Tensor& lu,
    const Tensor& pivs) {
  auto eps = at::native::_get_epsilon(c10::toRealValueType(self.scalar_type()));
  auto n = self.size(-1);
  auto eps_tensor = at::tensor(eps, self.options());
  auto condition_diagonal = [&](const Tensor& x) {
    auto x_diag = x.diagonal(0, -2, -1);
    auto x_diag_conditioned = at::where(x_diag == 0.0, eps_tensor, x_diag);
    x_diag.copy_(x_diag_conditioned);
  };

  // create a matrix d := (det_grad * det.conj()) I
  // NOTE: we do not use the shorter version
  // auto d = at::zeros_like(self);
  // d.diagonal(0, -2, -1).copy_((det_grad * det.conj()).unsqueeze(-1));
  // to avoid in-place operations to eliminate potential issues with Vmap
  auto det_expanded_sizes = det.sizes().vec();
  det_expanded_sizes.push_back(n);
  auto d_diag = det_grad * det.conj();
  auto d = at::diag_embed(d_diag.unsqueeze(-1).expand(det_expanded_sizes));
  // make sure that d is Fortran-contiguous. The transposition is sufficient as
  // d is a diagonal square matrix
  d = d.transpose(-2, -1);

  // we want to condition the diagonal of the lu Tensor, but it is not allowed
  // to modify arguments of backward functions in-place, hence the cloning.
  auto lu_clone = lu.clone();
  condition_diagonal(lu_clone);

  auto trans = self.is_complex() ? TransposeType::ConjTranspose
                                 : TransposeType::Transpose;
  auto infos_tensor = at::zeros(
      native::batchCount(d),
      self.options().dtype(kInt).device(DeviceType::CPU));
  std::vector<int32_t> infos(native::batchCount(d), 0);

  // d is modified in-place and will contain the result
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      d.scalar_type(), "_det_lu_based_helper_backward_helper", [&] {
        apply_lu_solve_xpu_<scalar_t>(d, lu_clone, pivs, infos, trans);
      });

  std::copy(
      infos.begin(), infos.end(), infos_tensor.template data_ptr<int32_t>());
  at::_linalg_check_errors(
      infos_tensor, "_det_lu_based_helper_backward_helper", self.dim() == 2);

  return d;
}

void lu_solve_mkl(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans) {
  std::vector<int32_t> infos_vec(native::batchCount(LU), 0);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU.scalar_type(), "lu_solve_xpu", [&] {
    apply_lu_solve_xpu_<scalar_t>(B, LU, pivots, infos_vec, trans);
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

  auto sizes = LU.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];

  // make column major strides for BLAS
  auto LU_strides = at::native::batched_matrix_contiguous_strides(
      sizes,
      /*f-contig*=*/true);
  // auto LU_new = set_strided(LU, sizes, LU_strides, LU.options());
  auto LU_new = at::empty_strided(sizes, LU_strides, LU.options());
  // Tensor LU_use = C10_UNLIKELY(LU_new.has_value()) ? LU_new.values() : LU;
  Tensor LU_use = LU;

  // Set sizes to the size of pivots
  sizes.pop_back();
  sizes.back() = std::min(m, n);
  pivots.contiguous();
  // set_contiguous_no_create(pivots, sizes, LU.options().dtype(kInt));

  // Set sizes to the size of info
  sizes.pop_back();
  // set_contiguous_no_create(info, sizes, LU.options().dtype(kInt));
  info.contiguous();

  TORCH_CHECK(
      pivot,
      "linalg.lu_factor: LU without pivoting is not implemented on the XPU");

  // handle the info
  std::vector<int32_t> infos_vec(native::batchCount(LU), 0);
  // mkl needs long for pivots, but PT is int
  Tensor pivots_ = at::empty(pivots.sizes(), pivots.options().dtype(kLong));
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(LU_use.scalar_type(), "lu_xpu", [&] {
    apply_lu_xpu_<scalar_t>(LU_use, pivots_, infos_vec);
  });
  auto expected_info_shape =
      IntArrayRef(LU_use.sizes().cbegin(), LU_use.sizes().cend() - 2);

  info.copy_(at::from_blob(
      (int32_t*)(infos_vec.data()),
      expected_info_shape,
      c10::toRealValueType(info.scalar_type())));

  // Copy to original pivots tensor
  pivots.copy_(pivots_);
  // if (LU_new.has_value())
  LU.copy_(LU_use);
  // return std::tuple<Tensor&, Tensor&, Tensor&>(LU, pivots, info);
}

} // namespace at::native::xpu
#endif // USE_ONEMKL
