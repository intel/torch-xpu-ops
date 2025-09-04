#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/Runtime.h>
#include <oneapi/mkl/blas.hpp>
#include <torch/library.h>

namespace at::native {

#if defined(USE_ONEMKL_XPU)

at::Tensor& handle_output_copy(at::Tensor& out, const at::Tensor& result) {
  if (!out.is_same(result)) {
    if (out.sizes() == result.sizes()) {
      out.copy_(result);
    } else {
      out.copy_(result.view(out.sizes()));
    }
  }

  return out;
}

template <typename T>
at::Tensor& mm_complex_out_xpu_impl(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  at::Tensor self_cont = self.contiguous().resolve_conj();
  at::Tensor mat2_cont = mat2.contiguous().resolve_conj();
  at::Tensor out_cont = out.contiguous().resolve_conj();

  const int64_t m = self_cont.sizes().at(0);
  const int64_t n = mat2_cont.sizes().at(1);
  const int64_t k = self_cont.sizes().at(1);

  constexpr std::complex<T> alpha = {T(1), T(0)};
  constexpr std::complex<T> beta = {T(0), T(0)};

  oneapi::mkl::blas::row_major::gemm(
      c10::xpu::getCurrentXPUStream().queue(),
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      m,
      n,
      k,
      alpha,
      reinterpret_cast<const std::complex<T>*>(self_cont.const_data_ptr()),
      k,
      reinterpret_cast<const std::complex<T>*>(mat2_cont.const_data_ptr()),
      n,
      beta,
      reinterpret_cast<std::complex<T>*>(out_cont.data_ptr()),
      n);

  return handle_output_copy(out, out_cont);
}

at::Tensor& mm_complex_out_xpu(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
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
at::Tensor& bmm_complex_out_xpu_impl(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  at::Tensor self_cont = self.contiguous().resolve_conj();
  at::Tensor mat2_cont = mat2.contiguous().resolve_conj();
  at::Tensor out_cont = out.contiguous().resolve_conj();

  const int64_t batch_size = self_cont.sizes().at(0);
  const int64_t m = self_cont.sizes().at(1);
  const int64_t n = mat2_cont.sizes().at(2);
  const int64_t k = self_cont.sizes().at(2);

  constexpr std::complex<T> alpha = {T(1), T(0)};
  constexpr std::complex<T> beta = {T(0), T(0)};

  oneapi::mkl::blas::row_major::gemm_batch(
      c10::xpu::getCurrentXPUStream().queue(),
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      m,
      n,
      k,
      alpha,
      reinterpret_cast<const std::complex<T>*>(self_cont.const_data_ptr()),
      k,
      m * k,
      reinterpret_cast<const std::complex<T>*>(mat2_cont.const_data_ptr()),
      n,
      k * n,
      beta,
      reinterpret_cast<std::complex<T>*>(out_cont.data_ptr()),
      n,
      m * n,
      batch_size);

  return handle_output_copy(out, out_cont);
}

at::Tensor& bmm_complex_out_xpu(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
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
at::Tensor& addmm_complex_out_xpu_impl(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  at::Tensor mat1_cont = mat1.contiguous().resolve_conj();
  at::Tensor mat2_cont = mat2.contiguous().resolve_conj();
  at::Tensor self_cont = self.contiguous().resolve_conj().clone().detach();

  const int64_t m = mat1_cont.sizes().at(0);
  const int64_t n = mat2_cont.sizes().at(1);
  const int64_t k = mat1_cont.sizes().at(1);

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

  const std::vector<int64_t> mm_output_size = {m, n};
  if (self_cont.sizes() != mm_output_size) {
    self_cont = at::broadcast_to(self_cont, mm_output_size).contiguous();
  }

  std::complex<T> complex_alpha =
      static_cast<std::complex<T>>(alpha.toComplexDouble());
  std::complex<T> complex_beta =
      static_cast<std::complex<T>>(beta.toComplexDouble());

  oneapi::mkl::blas::row_major::gemm(
      c10::xpu::getCurrentXPUStream().queue(),
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      m,
      n,
      k,
      complex_alpha,
      reinterpret_cast<const std::complex<T>*>(mat1_cont.const_data_ptr()),
      k,
      reinterpret_cast<const std::complex<T>*>(mat2_cont.const_data_ptr()),
      n,
      complex_beta,
      reinterpret_cast<std::complex<T>*>(self_cont.data_ptr()),
      n);

  return handle_output_copy(out, self_cont);
}

at::Tensor& addmm_complex_out_xpu(
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
at::Tensor& baddbmm_complex_out_xpu_impl(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  at::Tensor batch1_cont = batch1.contiguous().resolve_conj();
  at::Tensor batch2_cont = batch2.contiguous().resolve_conj();
  at::Tensor self_cont = self.contiguous().resolve_conj().clone().detach();

  const int64_t batch_size = batch1_cont.sizes().at(0);
  const int64_t m = batch1_cont.sizes().at(1);
  const int64_t n = batch2_cont.sizes().at(2);
  const int64_t k = batch1_cont.sizes().at(2);

  const std::vector<int64_t> mm_output_size = {batch_size, m, n};
  if (self_cont.sizes() != mm_output_size) {
    self_cont = at::broadcast_to(self_cont, mm_output_size).contiguous();
  }

  std::complex<T> complex_alpha =
      static_cast<std::complex<T>>(alpha.toComplexDouble());
  std::complex<T> complex_beta =
      static_cast<std::complex<T>>(beta.toComplexDouble());

  oneapi::mkl::blas::row_major::gemm_batch(
      c10::xpu::getCurrentXPUStream().queue(),
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      m,
      n,
      k,
      complex_alpha,
      reinterpret_cast<const std::complex<T>*>(batch1_cont.const_data_ptr()),
      k,
      m * k,
      reinterpret_cast<const std::complex<T>*>(batch2_cont.const_data_ptr()),
      n,
      k * n,
      complex_beta,
      reinterpret_cast<std::complex<T>*>(self_cont.data_ptr()),
      n,
      m * n,
      batch_size);

  return handle_output_copy(out, self_cont);
}

at::Tensor& baddbmm_complex_out_xpu(
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

#endif // USE_ONEMKL_XPU

} // namespace at::native
