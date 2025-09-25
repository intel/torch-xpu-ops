#include <ATen/ATen.h>
#include <ATen/native/xpu/mkl/BlasImpl.h>
#include <comm/Runtime.h>
#include <oneapi/mkl/blas.hpp>
#include <torch/library.h>

namespace at::native::xpu {

namespace {

inline bool is_column_major(
    const int64_t stride_first,
    const int64_t stride_second,
    const int64_t dim_first,
    const int64_t dim_second,
    const bool contiguous_batch,
    const bool check_dim_second = true) {
  return contiguous_batch && stride_first == 1 &&
      ((dim_second == 1 && check_dim_second) ||
       stride_second >= std::max(int64_t{1}, dim_first));
}

inline bool is_row_major(
    const int64_t stride_first,
    const int64_t stride_second,
    const int64_t dim_first,
    const int64_t dim_second,
    const bool contiguous_batch,
    const bool check_dim_first = true) {
  return contiguous_batch && stride_second == 1 &&
      ((dim_first == 1 && check_dim_first) ||
       stride_first >= std::max(int64_t{1}, dim_second));
}

std::pair<Tensor, bool> process_result_matrix(
    const Tensor& result,
    IntArrayRef result_sizes) {
  const auto result_strides = result.strides();
  const int64_t ndim = result_strides.size();
  const int64_t last_dim = ndim - 1;
  const int64_t second_last_dim = ndim - 2;

  const bool contiguous_batch = ndim > 2
      ? result_strides[0] == (result_sizes[1] * result_sizes[2])
      : true;

  Tensor c = result.resolve_conj();

  if (is_column_major(
          result_strides[second_last_dim],
          result_strides[last_dim],
          result_sizes[second_last_dim],
          result_sizes[last_dim],
          contiguous_batch)) {
    return {c, false};
  }

  if (is_row_major(
          result_strides[second_last_dim],
          result_strides[last_dim],
          result_sizes[second_last_dim],
          result_sizes[last_dim],
          contiguous_batch)) {
    return {c, true};
  }

  // Matrix is not contiguous - make it contiguous while preserving layout
  c = c.transpose(second_last_dim, last_dim)
          .contiguous()
          .transpose_(second_last_dim, last_dim);
  return {c, false};
}

std::pair<Tensor, bool> process_matrix(
    const Tensor& m,
    bool transpose_c,
    int64_t first_dim,
    int64_t second_dim) {
  const auto m_strides = m.strides();
  const int64_t ndim = m_strides.size();
  const int64_t last_stride = m_strides[ndim - 1];
  const int64_t second_last_stride = m_strides[ndim - 2];

  const bool contiguous_batch =
      ndim > 2 ? m_strides[0] == (m.sizes()[1] * m.sizes()[2]) : true;

  const int64_t stride_inner = transpose_c ? last_stride : second_last_stride;
  const int64_t stride_outer = transpose_c ? second_last_stride : last_stride;

  if (is_column_major(
          stride_inner,
          stride_outer,
          first_dim,
          second_dim,
          contiguous_batch,
          false)) {
    return {m.resolve_conj(), false};
  }

  if (is_row_major(
          stride_inner,
          stride_outer,
          first_dim,
          second_dim,
          contiguous_batch,
          false)) {
    return {m, true};
  }

  // Matrix needs to be made contiguous with transposition based on transpose_c
  return {m.clone(MemoryFormat::Contiguous), !transpose_c};
}

void copy_result_to_output(Tensor& output, const Tensor& result) {
  if (!output.is_same(result)) {
    if (output.sizes() == result.sizes()) {
      output.copy_(result);
    } else {
      output.copy_(result.view(output.sizes()));
    }
  }
}

inline oneapi::mkl::transpose get_transpose_type(
    const Tensor& matrix,
    const bool is_transposed) {
  return is_transposed
      ? matrix.is_conj() ? oneapi::mkl::transpose::C : oneapi::mkl::transpose::T
      : oneapi::mkl::transpose::N;
}

// for the corner case: result tensor with size [m, 1], stride [1, 1]
// we cannot use stride to get its leading dimension, whose value should be m.
inline int64_t get_ldc(const bool is_transposed, const Tensor& c) {
  int64_t ldc;
  const int64_t ndim = c.dim();

  // Handle the corner case where the last two strides are both 1
  if (c.strides()[ndim - 2] == c.strides()[ndim - 1] &&
      c.strides()[ndim - 1] == 1) {
    ldc = c.sizes()[is_transposed ? ndim - 1 : ndim - 2];
  } else {
    ldc = c.strides()[is_transposed ? ndim - 2 : ndim - 1];
  }
  return ldc;
}

inline int64_t get_stridec(const Tensor c) {
  return c.sizes()[1] * c.sizes()[2];
}

template <typename T>
void perform_blas_matmul(
    Tensor& out,
    std::optional<Tensor> self,
    const Tensor& mat1,
    const Tensor& mat2,
    const std::complex<T> alpha = {T(1), T(0)},
    const std::complex<T> beta = {T(0), T(0)}) {
  Tensor& result = self.has_value() ? self.value() : out;

  const int64_t ndim = mat1.dim();

  const auto result_sizes = result.sizes();
  auto [c, transpose_c] = process_result_matrix(result, result_sizes);
  // For cases when C matrix is transposed we need to switch m1 and m2 to use
  // column_major implementation.
  const Tensor& m1 = transpose_c ? mat2 : mat1;
  const Tensor& m2 = transpose_c ? mat1 : mat2;

  const int64_t m = result_sizes[transpose_c ? ndim - 1 : ndim - 2];
  const int64_t n = result_sizes[transpose_c ? ndim - 2 : ndim - 1];
  const int64_t k = mat1.sizes()[ndim - 1];

  auto [a, transpose_a] = process_matrix(m1, transpose_c, m, k);
  auto [b, transpose_b] = process_matrix(m2, transpose_c, k, n);

  const int64_t lda =
      a.strides()[(transpose_a == transpose_c) ? ndim - 1 : ndim - 2];
  const int64_t ldb =
      b.strides()[(transpose_b == transpose_c) ? ndim - 1 : ndim - 2];
  const int64_t ldc = get_ldc(transpose_c, c);

  const std::complex<T>* A =
      reinterpret_cast<const std::complex<T>*>(a.const_data_ptr());
  const std::complex<T>* B =
      reinterpret_cast<const std::complex<T>*>(b.const_data_ptr());
  std::complex<T>* C = reinterpret_cast<std::complex<T>*>(c.data_ptr());
  auto queue = c10::xpu::getCurrentXPUStream().queue();

  const oneapi::mkl::transpose transA = get_transpose_type(a, transpose_a);
  const oneapi::mkl::transpose transB = get_transpose_type(b, transpose_b);

  if (ndim == 2) {
    oneapi::mkl::blas::column_major::gemm(
        queue, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    const int64_t stridea = a.strides()[0];
    const int64_t strideb = b.strides()[0];
    const int64_t num_batch = c.sizes()[0];

    oneapi::mkl::blas::column_major::gemm_batch(
        queue,
        transA,
        transB,
        m,
        n,
        k,
        alpha,
        A,
        lda,
        stridea,
        B,
        ldb,
        strideb,
        beta,
        C,
        ldc,
        get_stridec(c),
        num_batch);
  }

  copy_result_to_output(out, c);
}

Tensor prepare_result_tensor(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    bool is_batched = false) {
  Tensor result = self.contiguous().resolve_conj().clone().detach();

  std::vector<int64_t> expected_output_size = is_batched
      ? std::vector<int64_t>{mat1.size(0), mat1.size(1), mat2.size(2)}
      : std::vector<int64_t>{mat1.size(0), mat2.size(1)};

  if (result.sizes() != expected_output_size) {
    result = broadcast_to(result, expected_output_size).contiguous();
  }

  return result;
}

} // namespace

Tensor& mm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "mm_complex_out_xpu_mkl", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    perform_blas_matmul<underlying_t>(out, std::nullopt, self, mat2);
  });

  return out;
}

Tensor& bmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "bmm_complex_out_xpu_mkl", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    perform_blas_matmul<underlying_t>(out, std::nullopt, self, mat2);
  });

  return out;
}

Tensor& addmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  AT_DISPATCH_COMPLEX_TYPES(
      self.scalar_type(), "addmm_complex_out_xpu_mkl", [&] {
        using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
        perform_blas_matmul<underlying_t>(
            out,
            prepare_result_tensor(self, mat1, mat2, false),
            mat1,
            mat2,
            static_cast<std::complex<underlying_t>>(alpha.toComplexDouble()),
            static_cast<std::complex<underlying_t>>(beta.toComplexDouble()));
      });

  return out;
}

Tensor& baddbmm_complex_out_xpu_mkl(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  AT_DISPATCH_COMPLEX_TYPES(
      self.scalar_type(), "baddbmm_complex_out_xpu_mkl", [&] {
        using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
        perform_blas_matmul<underlying_t>(
            out,
            prepare_result_tensor(self, batch1, batch2, true),
            batch1,
            batch2,
            static_cast<std::complex<underlying_t>>(alpha.toComplexDouble()),
            static_cast<std::complex<underlying_t>>(beta.toComplexDouble()));
      });

  return out;
}
} // namespace at::native::xpu
