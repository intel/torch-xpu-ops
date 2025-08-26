#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/mkldnn/xpu/Blas.h>
#include <ATen/xpu/XPUContext.h>
#include <comm/Runtime.h>
#include <oneapi/mkl/blas.hpp>
#include <torch/library.h>

namespace at::native {

inline at::Tensor resolveViewsAndConjugation(const at::Tensor& input) {
  at::Tensor input_resolved = input.is_conj() ? input.resolve_conj() : input;
  at::Tensor input_contiguous = input_resolved.is_contiguous()
      ? input_resolved
      : input_resolved.contiguous();

  return input_contiguous;
}

template <typename T>
at::Tensor& mm_complex_out_xpu_impl(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  at::Tensor self_cont = resolveViewsAndConjugation(self);
  at::Tensor mat2_cont = resolveViewsAndConjugation(mat2);
  at::Tensor out_cont = resolveViewsAndConjugation(out);

  const int64_t m = self_cont.sizes().at(0);
  const int64_t n = mat2_cont.sizes().at(1);
  const int64_t k = self_cont.sizes().at(1);

  constexpr std::complex<T> alpha = {T(1.0), T(0.0)};
  constexpr std::complex<T> beta = {T(0.0), T(0.0)};

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

  if (!out.is_same(out_cont)) {
    out.copy_(out_cont);
  }

  return out;
}

at::Tensor& mm_complex_out_xpu(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  at::Tensor out_ref = at::mm(self.cpu(), mat2.cpu());

  AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "mm_complex_out_xpu", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    mm_complex_out_xpu_impl<underlying_t>(self, mat2, out);
  });

  return out;
}

template <typename T>
at::Tensor& bmm_complex_out_xpu_impl(
    const at::Tensor& self,
    const at::Tensor& batch2,
    at::Tensor& out) {
  at::Tensor self_cont = resolveViewsAndConjugation(self);
  at::Tensor batch2_cont = resolveViewsAndConjugation(batch2);
  at::Tensor out_cont = resolveViewsAndConjugation(out);

  const int64_t batch_size = self_cont.sizes().at(0);
  const int64_t m = self_cont.sizes().at(1);
  const int64_t n = batch2_cont.sizes().at(2);
  const int64_t k = self_cont.sizes().at(2);

  constexpr std::complex<T> alpha = {T(1.0f), T(0.0f)};
  constexpr std::complex<T> beta = {T(0.0f), T(0.0f)};

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
      reinterpret_cast<const std::complex<T>*>(batch2_cont.const_data_ptr()),
      n,
      k * n,
      beta,
      reinterpret_cast<std::complex<T>*>(out_cont.data_ptr()),
      n,
      m * n,
      batch_size);

  if (!out.is_same(out_cont)) {
    out.copy_(out_cont);
  }

  return out;
}

at::Tensor& bmm_complex_out_xpu(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {

  AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "bmm_complex_out_xpu", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    bmm_complex_out_xpu_impl<underlying_t>(self, mat2, out);
  });

  return out;
}

template <typename T>
at::Tensor& addmm_complex_out_xpu_impl(
    const Tensor& input,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  at::Tensor mat1_cont = resolveViewsAndConjugation(mat1);
  at::Tensor mat2_cont = resolveViewsAndConjugation(mat2);
  at::Tensor input_cont = resolveViewsAndConjugation(input).clone().detach();

  const int64_t m = mat1_cont.sizes().at(0);
  const int64_t n = mat2_cont.sizes().at(1);
  const int64_t k = mat1_cont.sizes().at(1);

  // Some paths in the code below do not handle multiplications of the form [n, 0] x [0, m]
  if (k == 0) {
    if (result.numel() == 0) {
      return result;
    }
    if (beta.toComplexDouble() == 0.0) {
      result.zero_();
    } else {
      if (!input.is_same(result)) {
        result.copy_(input);
      }
      result.mul_(beta);
    }
    return result;
  }

  if (m == 0 || n == 0) {
    return result;
  }

  const std::vector<int64_t> mm_output_size = {m, n};
  if (input_cont.sizes() != mm_output_size) {
    input_cont = at::broadcast_to(input_cont, mm_output_size).contiguous();
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
      reinterpret_cast<std::complex<T>*>(input_cont.data_ptr()),
      n);

  if (result.sizes() == input_cont.sizes()) {
    result.copy_(input_cont);
  } else {
    result.copy_(input_cont.view(result.sizes()));
  }

  return result;
}

at::Tensor& addmm_complex_out_xpu(
    const Tensor& input,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {

  AT_DISPATCH_COMPLEX_TYPES(input.scalar_type(), "addmm_complex_out_xpu", [&] {
    using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
    addmm_complex_out_xpu_impl<underlying_t>(
        input, mat1, mat2, beta, alpha, result);
  });

  return result;
}

template <typename T>
at::Tensor& baddbmm_complex_out_xpu_impl(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  at::Tensor batch1_cont = resolveViewsAndConjugation(batch1);
  at::Tensor batch2_cont = resolveViewsAndConjugation(batch2);
  at::Tensor input_cont = resolveViewsAndConjugation(input).clone().detach();

  const int64_t batch_size = batch1_cont.sizes().at(0);
  const int64_t m = batch1_cont.sizes().at(1);
  const int64_t n = batch2_cont.sizes().at(2);
  const int64_t k = batch1_cont.sizes().at(2);

  const std::vector<int64_t> mm_output_size = {batch_size, m, n};
  if (input_cont.sizes() != mm_output_size) {
    input_cont = at::broadcast_to(input_cont, mm_output_size).contiguous();;
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
      reinterpret_cast<std::complex<T>*>(input_cont.data_ptr()),
      n,
      m * n,
      batch_size);

  if (result.sizes() == input_cont.sizes()) {
    result.copy_(input_cont);
  } else {
    result.copy_(input_cont.view(result.sizes()));
  }

  return result;
}

at::Tensor& baddbmm_complex_out_xpu(
    const Tensor& input,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {

  AT_DISPATCH_COMPLEX_TYPES(
      input.scalar_type(), "baddbmm_complex_out_xpu", [&] {
        using underlying_t = typename c10::scalar_value_type<scalar_t>::type;
        baddbmm_complex_out_xpu_impl<underlying_t>(
            input, batch1, batch2, beta, alpha, result);
      });

  return result;
}

REGISTER_XPU_DISPATCH(mm_complex_stub, &mm_complex_out_xpu)
REGISTER_XPU_DISPATCH(bmm_complex_stub, &bmm_complex_out_xpu)
REGISTER_XPU_DISPATCH(addmm_complex_stub, &addmm_complex_out_xpu)
REGISTER_XPU_DISPATCH(baddbmm_complex_stub, &baddbmm_complex_out_xpu)

} // namespace at::native