#include <ATen/ATen.h>
#include <comm/Runtime.h>
#include <oneapi/mkl/blas.hpp>
#include <torch/library.h>

namespace at::native {

#if defined(USE_ONEMKL_XPU)

std::pair<Tensor, bool> process_result_matrix(
    const Tensor& result,
    IntArrayRef result_sizes) {
  const auto result_strides = result.strides();
  const int64_t ndim = result_strides.size();
  const int64_t last_dim = ndim - 1;
  const int64_t second_last_dim = ndim - 2;

  Tensor c = result.resolve_conj();

  // Check if already in column-major layout (first dimension has unit stride)
  if (result_strides[second_last_dim] == 1 &&
      (result_sizes[last_dim] == 1 ||
       result_strides[last_dim] ==
           std::max(int64_t{1}, result_sizes[second_last_dim]))) {
    return {c, false};
  }

  // Check if in row-major layout (second dimension has unit stride)
  if (result_strides[last_dim] == 1 &&
      (result_sizes[second_last_dim] == 1 ||
       result_strides[second_last_dim] ==
           std::max(int64_t{1}, result_sizes[last_dim]))) {
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

  const int64_t stride_inner = transpose_c ? last_stride : second_last_stride;
  const int64_t stride_outer = transpose_c ? second_last_stride : last_stride;

  // Check if matrix is already in the preferred layout (column-major for BLAS)
  if (stride_inner == 1 && stride_outer == std::max(int64_t{1}, first_dim)) {
    return {m.resolve_conj(), false};
  }

  // Check if matrix needs transposition but has unit stride in the other
  // dimension
  if (stride_outer == 1 && stride_inner == std::max(int64_t{1}, second_dim)) {
    return {m, true};
  }

  // Matrix needs to be made contiguous with transposition based on transpose_c
  return {m.clone(MemoryFormat::Contiguous), !transpose_c};
}

Tensor& copy_result_to_output(Tensor& output, const Tensor& result) {
  if (!output.is_same(result)) {
    if (output.sizes() == result.sizes()) {
      output.copy_(result);
    } else {
      output.copy_(result.view(output.sizes()));
    }
  }

  return output;
}

static inline oneapi::mkl::transpose get_transpose_type(
    const Tensor& matrix,
    const bool is_transposed) {
  return is_transposed
      ? matrix.is_conj() ? oneapi::mkl::transpose::C : oneapi::mkl::transpose::T
      : oneapi::mkl::transpose::N;
}

// for the corner case: result tensor with size [m, 1], stride [1, 1]
// we cannot use stride to get its leading dimension, whose value should be m.
static inline int64_t get_ldc(const bool is_transposed, const Tensor& c) {
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

static inline int64_t get_stridec(const Tensor c) {
  return c.sizes()[1] * c.sizes()[2];
}

template <typename T>
static void perform_blas_matmul(
    Tensor& c,
    const Tensor& a,
    const Tensor& b,
    const bool transpose_a,
    const bool transpose_b,
    const bool transpose_c,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const std::complex<T> alpha = {T(1), T(0)},
    const std::complex<T> beta = {T(0), T(0)}) {
  const int64_t ndim = a.dim();

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

  if (c.dim() == 2) {
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
}

static Tensor prepare_result_tensor(
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

template <typename T>
Tensor& mm_complex_out_xpu_impl(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& result) {
  if (result.numel() == 0) {
    return result;
  }

  const auto result_sizes = result.sizes();
  auto [c, transpose_c] = process_result_matrix(result, result_sizes);
  // For cases when C matrix is transposed we need to switch m1 and m2 to use
  // column_major implementation.
  const Tensor& m1 = transpose_c ? mat2 : self;
  const Tensor& m2 = transpose_c ? self : mat2;

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = self.sizes()[1];

  auto [a, transpose_a] = process_matrix(m1, transpose_c, m, k);
  auto [b, transpose_b] = process_matrix(m2, transpose_c, k, n);

  perform_blas_matmul<T>(
      c, a, b, transpose_a, transpose_b, transpose_c, m, n, k);

  return copy_result_to_output(result, c);
}

Tensor& mm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(), "_mm_mkl.out expects self to be a complex datatype.");

  AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "mm_complex_out_xpu", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    mm_complex_out_xpu_impl<underlying_t>(self, mat2, out);
  });

  return out;
}

template <typename T>
Tensor& bmm_complex_out_xpu_impl(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& result) {
  const auto result_sizes = result.sizes();
  auto [c, transpose_c] = process_result_matrix(result, result_sizes);
  const Tensor& batch1 = transpose_c ? mat2 : self;
  const Tensor& batch2 = transpose_c ? self : mat2;

  const int64_t m = result_sizes[transpose_c ? 2 : 1];
  const int64_t n = result_sizes[transpose_c ? 1 : 2];
  const int64_t k = batch1.sizes()[transpose_c ? 1 : 2];

  auto [a, transpose_a] = process_matrix(batch1, transpose_c, m, k);
  auto [b, transpose_b] = process_matrix(batch2, transpose_c, k, n);

  perform_blas_matmul<T>(
      c, a, b, transpose_a, transpose_b, transpose_c, m, n, k);

  return copy_result_to_output(result, c);
}

Tensor& bmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(), "_bmm_mkl.out expects self to be a complex datatype.");

  AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "bmm_complex_out_xpu", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    bmm_complex_out_xpu_impl<underlying_t>(self, mat2, out);
  });

  return out;
}

template <typename T>
Tensor& addmm_complex_out_xpu_impl(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  Tensor result = prepare_result_tensor(self, mat1, mat2, false);
  if (result.numel() == 0) {
    return out;
  }

  const auto result_sizes = result.sizes();
  auto [c, transpose_c] = process_result_matrix(result, result_sizes);
  const Tensor& m1 = transpose_c ? mat2 : mat1;
  const Tensor& m2 = transpose_c ? mat1 : mat2;

  const int64_t m = result_sizes[transpose_c ? 1 : 0];
  const int64_t n = result_sizes[transpose_c ? 0 : 1];
  const int64_t k = m1.sizes()[transpose_c ? 0 : 1];

  if (k == 0) {
    if (out.numel() == 0) {
      return out;
    }
    if (beta.toComplexDouble() == 0.0) {
      out.zero_();
    } else {
      if (!self.is_same(out)) {
        out.copy_(self);
      }
      out.mul_(beta);
    }
    return out;
  }

  if (m == 0 || n == 0) {
    return out;
  }

  auto [a, transpose_a] = process_matrix(m1, transpose_c, m, k);
  auto [b, transpose_b] = process_matrix(m2, transpose_c, k, n);

  perform_blas_matmul(
      c,
      a,
      b,
      transpose_a,
      transpose_b,
      transpose_c,
      m,
      n,
      k,
      static_cast<std::complex<T>>(alpha.toComplexDouble()),
      static_cast<std::complex<T>>(beta.toComplexDouble()));

  return copy_result_to_output(out, c);
}

Tensor& addmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(),
      "_addmm_mkl.out expects self to be a complex datatype.");

  AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "addmm_complex_out_xpu", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    addmm_complex_out_xpu_impl<underlying_t>(
        self, mat1, mat2, beta, alpha, out);
  });

  return out;
}

template <typename T>
Tensor& baddbmm_complex_out_xpu_impl(
    const Tensor& self,
    const Tensor& batch1_input,
    const Tensor& batch2_input,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  Tensor result = prepare_result_tensor(self, batch1_input, batch2_input, true);
  const auto result_sizes = result.sizes();

  if (result.numel() == 0) {
    return out;
  } else if (batch1_input.sizes()[2] == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      result.zero_();
    }
  }

  auto [c, transpose_c] = process_result_matrix(result, result_sizes);
  const Tensor& batch1 = transpose_c ? batch2_input : batch1_input;
  const Tensor& batch2 = transpose_c ? batch1_input : batch2_input;

  const int64_t m = result_sizes[transpose_c ? 2 : 1];
  const int64_t n = result_sizes[transpose_c ? 1 : 2];
  const int64_t k = batch1.sizes()[transpose_c ? 1 : 2];

  auto [a, transpose_a] = process_matrix(batch1, transpose_c, m, k);
  auto [b, transpose_b] = process_matrix(batch2, transpose_c, k, n);

  perform_blas_matmul(
      c,
      a,
      b,
      transpose_a,
      transpose_b,
      transpose_c,
      m,
      n,
      k,
      static_cast<std::complex<T>>(alpha.toComplexDouble()),
      static_cast<std::complex<T>>(beta.toComplexDouble()));

  return copy_result_to_output(out, c);
}

Tensor& baddbmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(),
      "_baddbmm_mkl.out expects self to be a complex datatype.");

  AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "baddbmm_complex_out_xpu", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    baddbmm_complex_out_xpu_impl<underlying_t>(
        self, batch1, batch2, beta, alpha, out);
  });

  return out;
}

#endif // USE_ONEMKL_XPU

TORCH_LIBRARY_FRAGMENT(aten, m) {
  m.def(
      "aten::_mm_mkl.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "aten::_bmm_mkl.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "aten::_addmm_mkl.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "aten::_baddbmm_mkl.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)");
}

#if defined(USE_ONEMKL_XPU)

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("aten::_mm_mkl.out", mm_complex_out_xpu);
  m.impl("aten::_bmm_mkl.out", bmm_complex_out_xpu);
  m.impl("aten::_addmm_mkl.out", addmm_complex_out_xpu);
  m.impl("aten::_baddbmm_mkl.out", baddbmm_complex_out_xpu);
}

// Conjugated tensors are handled inside kernel and for some cases there is no need to resolve conjugation which improves performance.
TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("aten::_mm_mkl.out", torch::CppFunction::makeFallthrough());
  m.impl("aten::_bmm_mkl.out", torch::CppFunction::makeFallthrough());
  m.impl("aten::_addmm_mkl.out", torch::CppFunction::makeFallthrough());
  m.impl("aten::_baddbmm_mkl.out", torch::CppFunction::makeFallthrough());
}

#endif // USE_ONEMKL_XPU

} // namespace at::native
