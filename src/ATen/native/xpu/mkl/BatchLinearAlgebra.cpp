#if defined(USE_ONEMKL)

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>
#include <ATen/ops/_linalg_check_errors.h>

#include <comm/SYCLContext.h>
#include <comm/TensorInfo.h>
#include <oneapi/mkl/lapack.hpp>

namespace at::native::xpu {

namespace impl {

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
#ifdef USE_ONEMKL
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
#else
  AT_ERROR("svd: oneMKL library not found in compilation");
#endif
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
#ifdef USE_ONEMKL
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
#else
  AT_ERROR("svd: oneMKL library not found in compilation");
#endif
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
#ifdef USE_ONEMKL
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
#else
  AT_ERROR("svd: oneMKL library not found in compilation");
#endif
}

} // namespace impl

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
      impl::_create_U_S_VT(self, some, compute_uv);

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
      impl::apply_svd<scalar_t, value_t>(
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

  std::tie(U_tmp, S_tmp, Vh_tmp) = _svd_helper(A, some, /*compute_uv=*/true);
  svd_resize_and_copy("U", U_tmp, U);
  svd_resize_and_copy("S", S_tmp, S);
  svd_resize_and_copy("V", Vh_tmp, Vh);
}

} // namespace at::native::xpu
#endif // USE_ONEMKL
