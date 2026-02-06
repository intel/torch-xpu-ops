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
#include <ATen/native/ComplexHelper.h>
#include <ATen/native/xpu/Blas.h>
#if defined(USE_ONEMKL_XPU)
#include <ATen/native/xpu/mkl/BlasImpl.h>
#endif
#include <torch/library.h>

namespace at::native {

namespace {

class ConjPhysicalGuard final {
 public:
  explicit ConjPhysicalGuard(Tensor& out) : out_(out), was_conj_(out.is_conj()) {
    if (was_conj_) {
      out_.conj_physical_();
    }
  }

  ~ConjPhysicalGuard() {
    if (was_conj_) {
      out_.conj_physical_();
    }
  }

  Tensor real() const {
    return at::native::_view_as_real_physical(out_);
  }

 private:
  Tensor& out_;
  bool was_conj_;
};

// Implement complex mm using real GEMM decomposition
// Uses Gauss-Strassen optimization: 3 GEMMs instead of 4
Tensor& mm_complex_fallback(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  TORCH_WARN_ONCE(
      "Complex matrix multiplication is using fallback implementation. "
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto self_real = at::view_as_real(self.resolve_conj());
  auto mat2_real = at::view_as_real(mat2.resolve_conj());

  auto A_r = self_real.select(-1, 0);
  auto A_i = self_real.select(-1, 1);
  auto B_r = mat2_real.select(-1, 0);
  auto B_i = mat2_real.select(-1, 1);

  // P1 = A_r @ B_r, P2 = A_i @ B_i, P3 = (A_r + A_i) @ (B_r + B_i)
  // out_r = P1 - P2, out_i = P3 - P1 - P2
  auto P1 = at::mm(A_r, B_r);
  auto P2 = at::mm(A_i, B_i);
  auto P3 = at::mm(A_r + A_i, B_r + B_i);

  auto out_real = at::view_as_real(out);
  out_real.select(-1, 0).copy_(P1 - P2);
  out_real.select(-1, 1).copy_(P3 - P1 - P2);
  return out;
}

// Implement complex bmm using real GEMM decomposition
Tensor& bmm_complex_fallback(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  TORCH_WARN_ONCE(
      "Complex batch matrix multiplication is using fallback implementation. "
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto self_real = at::view_as_real(self.resolve_conj());
  auto mat2_real = at::view_as_real(mat2.resolve_conj());

  auto A_r = self_real.select(-1, 0);
  auto A_i = self_real.select(-1, 1);
  auto B_r = mat2_real.select(-1, 0);
  auto B_i = mat2_real.select(-1, 1);

  auto P1 = at::bmm(A_r, B_r);
  auto P2 = at::bmm(A_i, B_i);
  auto P3 = at::bmm(A_r + A_i, B_r + B_i);

  auto out_real = at::view_as_real(out);
  out_real.select(-1, 0).copy_(P1 - P2);
  out_real.select(-1, 1).copy_(P3 - P1 - P2);
  return out;
}

// Implement complex addmm using real GEMM decomposition
// Note: also called from addmv_out with mixed shapes (self may be scalar/1D,
// mat2 may be (N,1), out may be 1D). Must handle shape mismatches.
Tensor& addmm_complex_fallback(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  TORCH_WARN_ONCE(
      "Complex addmm is using fallback implementation. "
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto beta_c = beta.toComplexDouble();
  auto alpha_c = alpha.toComplexDouble();
  double beta_r = beta_c.real(), beta_i = beta_c.imag();
  double alpha_r = alpha_c.real(), alpha_i = alpha_c.imag();
  bool beta_zero = (beta_r == 0.0 && beta_i == 0.0);
  bool alpha_zero = (alpha_r == 0.0 && alpha_i == 0.0);

  // When beta==0, self may contain uninitialized/NaN values
  //  0 * NaN = NaN, so we must skip the self term entirely. Similarly skip matmul for alpha==0.
  if (beta_zero && alpha_zero) {
    out.zero_();
    return out;
  }

  // Only process self when beta != 0
  Tensor C_r, C_i;
  if (!beta_zero) {
    auto self_real = at::view_as_real(self.resolve_conj());
    C_r = self_real.select(-1, 0).contiguous();
    C_i = self_real.select(-1, 1).contiguous();
  }

  // Only process mat1/mat2 when alpha != 0
  Tensor A_r, A_i, B_r, B_i;
  if (!alpha_zero) {
    auto mat1_real = at::view_as_real(mat1.resolve_conj());
    auto mat2_real = at::view_as_real(mat2.resolve_conj());
    A_r = mat1_real.select(-1, 0);
    A_i = mat1_real.select(-1, 1);
    B_r = mat2_real.select(-1, 0);
    B_i = mat2_real.select(-1, 1);
  }

  // Compute result = beta * self + alpha * (mat1 @ mat2)
  // Prepare output views
  ConjPhysicalGuard out_guard(out);
  auto out_real = out_guard.real();

  if (alpha_zero) {
    // alpha == 0: result = beta * self (beta != 0 guaranteed by early return)
    // Handle reshape for addmv case
    auto result_r = beta_r * C_r - beta_i * C_i;
    auto result_i = beta_r * C_i + beta_i * C_r;
    out_real.select(-1, 0).copy_(result_r.reshape_as(out_real.select(-1, 0)));
    out_real.select(-1, 1).copy_(result_i.reshape_as(out_real.select(-1, 1)));
  } else {
    // Gauss-Strassen: 3 GEMMs for A @ B
    auto P1 = at::mm(A_r, B_r);
    auto P2 = at::mm(A_i, B_i);
    auto P3 = at::mm(A_r + A_i, B_r + B_i);
    auto AB_r = P1 - P2;
    auto AB_i = P3 - P1 - P2;

    if (beta_zero) {
      // beta == 0: result = alpha * (A@B)
      auto result_r = alpha_r * AB_r - alpha_i * AB_i;
      auto result_i = alpha_r * AB_i + alpha_i * AB_r;
      out_real.select(-1, 0).copy_(result_r.reshape_as(out_real.select(-1, 0)));
      out_real.select(-1, 1).copy_(result_i.reshape_as(out_real.select(-1, 1)));
    } else {
      // General case: result = beta*C + alpha*(A@B)
      auto result_r = (beta_r * C_r - beta_i * C_i) + (alpha_r * AB_r - alpha_i * AB_i);
      auto result_i = (beta_r * C_i + beta_i * C_r) + (alpha_r * AB_i + alpha_i * AB_r);
      out_real.select(-1, 0).copy_(result_r.reshape_as(out_real.select(-1, 0)));
      out_real.select(-1, 1).copy_(result_i.reshape_as(out_real.select(-1, 1)));
    }
  }
  return out;
}

// Implement complex baddbmm using real GEMM decomposition
Tensor& baddbmm_complex_fallback(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  TORCH_WARN_ONCE(
      "Complex baddbmm is using fallback implementation. "
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto beta_c = beta.toComplexDouble();
  auto alpha_c = alpha.toComplexDouble();
  double beta_r = beta_c.real(), beta_i = beta_c.imag();
  double alpha_r = alpha_c.real(), alpha_i = alpha_c.imag();
  bool beta_zero = (beta_r == 0.0 && beta_i == 0.0);
  bool alpha_zero = (alpha_r == 0.0 && alpha_i == 0.0);

  // When beta==0, self may contain uninitialized/NaN values.
  // 0 * NaN = NaN, so skip the self term entirely.
  if (beta_zero && alpha_zero) {
    out.zero_();
    return out;
  }

  // Only process self when beta != 0
  Tensor C_r, C_i;
  if (!beta_zero) {
    auto self_real = at::view_as_real(self.resolve_conj());
    C_r = self_real.select(-1, 0).contiguous();
    C_i = self_real.select(-1, 1).contiguous();
  }

  // Only process batch1/batch2 when alpha != 0
  Tensor A_r, A_i, B_r, B_i;
  if (!alpha_zero) {
    auto batch1_real = at::view_as_real(batch1.resolve_conj());
    auto batch2_real = at::view_as_real(batch2.resolve_conj());
    A_r = batch1_real.select(-1, 0);
    A_i = batch1_real.select(-1, 1);
    B_r = batch2_real.select(-1, 0);
    B_i = batch2_real.select(-1, 1);
  }

  // Compute result = beta * self + alpha * (batch1 @ batch2)
  ConjPhysicalGuard out_guard(out);
  auto out_real = out_guard.real();

  if (alpha_zero) {
    // alpha == 0: result = beta * self
    out_real.select(-1, 0).copy_(beta_r * C_r - beta_i * C_i);
    out_real.select(-1, 1).copy_(beta_r * C_i + beta_i * C_r);
  } else {
    // Gauss-Strassen: 3 GEMMs for A @ B
    auto P1 = at::bmm(A_r, B_r);
    auto P2 = at::bmm(A_i, B_i);
    auto P3 = at::bmm(A_r + A_i, B_r + B_i);
    auto AB_r = P1 - P2;
    auto AB_i = P3 - P1 - P2;

    if (beta_zero) {
      // beta == 0: result = alpha * (A@B)
      out_real.select(-1, 0).copy_(alpha_r * AB_r - alpha_i * AB_i);
      out_real.select(-1, 1).copy_(alpha_r * AB_i + alpha_i * AB_r);
    } else {
      // General case: result = beta*C + alpha*(A@B)
      out_real.select(-1, 0).copy_(
          (beta_r * C_r - beta_i * C_i) + (alpha_r * AB_r - alpha_i * AB_i));
      out_real.select(-1, 1).copy_(
          (beta_r * C_i + beta_i * C_r) + (alpha_r * AB_i + alpha_i * AB_r));
    }
  }
  return out;
}

} // anonymous namespace

Tensor& mm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(),
      "mm_complex_out_xpu expects self to be a complex datatype.");

  if (out.numel() == 0) {
    return out;
  }

#if defined(USE_ONEMKL_XPU)
  return at::native::xpu::mm_complex_out_xpu_mkl(self, mat2, out);
#else
  return mm_complex_fallback(self, mat2, out);
#endif // USE_ONEMKL_XPU
}

Tensor& bmm_complex_out_xpu(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  c10::DeviceGuard guard(self.device());
  TORCH_CHECK(
      self.is_complex(),
      "bmm_complex_out_xpu expects self to be a complex datatype.");

  if (out.numel() == 0) {
    return out;
  }

#if defined(USE_ONEMKL_XPU)
  return at::native::xpu::bmm_complex_out_xpu_mkl(self, mat2, out);
#else
  return bmm_complex_fallback(self, mat2, out);
#endif // USE_ONEMKL_XPU
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
      "addmm_complex_out_xpu expects self to be a complex datatype.");

  if (out.numel() == 0) {
    return out;
  }
  if (mat1.numel() == 0) {
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

#if defined(USE_ONEMKL_XPU)
  return at::native::xpu::addmm_complex_out_xpu_mkl(
      self, mat1, mat2, beta, alpha, out);
#else
  return addmm_complex_fallback(self, mat1, mat2, beta, alpha, out);
#endif // USE_ONEMKL_XPU
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
      "baddbmm_complex_out_xpu expects self to be a complex datatype.");

  if (out.numel() == 0) {
    return out;
  }
  if (batch1.numel() == 0) {
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

#if defined(USE_ONEMKL_XPU)
  return at::native::xpu::baddbmm_complex_out_xpu_mkl(
      self, batch1, batch2, beta, alpha, out);
#else
  return baddbmm_complex_fallback(self, batch1, batch2, beta, alpha, out);
#endif // USE_ONEMKL_XPU
}

} // namespace at::native
