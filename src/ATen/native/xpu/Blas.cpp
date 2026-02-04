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
#include <ATen/native/xpu/Blas.h>
#if defined(USE_ONEMKL_XPU)
#include <ATen/native/xpu/mkl/BlasImpl.h>
#elif AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/vdot_native.h>
#include <ATen/ops/dot_native.h>
#endif
#include <torch/library.h>

namespace at::native {

namespace {

// Implement complex mm using real GEMM decomposition
// Uses Gauss-Strassen optimization: 3 GEMMs instead of 4
Tensor& mm_complex_fallback(
    const Tensor& self,
    const Tensor& mat2,
    Tensor& out) {
  TORCH_WARN_ONCE(
      "Complex matrix multiplication is using fallback implementation."
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto self_real = at::view_as_real(self);
  auto mat2_real = at::view_as_real(mat2);

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
      "Complex batch matrix multiplication is using fallback implementation."
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto self_real = at::view_as_real(self);
  auto mat2_real = at::view_as_real(mat2);

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
Tensor& addmm_complex_fallback(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  TORCH_WARN_ONCE(
      "Complex addmm is using fallback implementation with real GEMM."
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto beta_c = beta.toComplexDouble();
  auto alpha_c = alpha.toComplexDouble();
  double beta_r = beta_c.real(), beta_i = beta_c.imag();
  double alpha_r = alpha_c.real(), alpha_i = alpha_c.imag();

  auto self_real = at::view_as_real(self);
  auto mat1_real = at::view_as_real(mat1);
  auto mat2_real = at::view_as_real(mat2);

  auto C_r = self_real.select(-1, 0);
  auto C_i = self_real.select(-1, 1);
  auto A_r = mat1_real.select(-1, 0);
  auto A_i = mat1_real.select(-1, 1);
  auto B_r = mat2_real.select(-1, 0);
  auto B_i = mat2_real.select(-1, 1);

  // Gauss-Strassen: 3 GEMMs for A @ B
  auto P1 = at::mm(A_r, B_r);
  auto P2 = at::mm(A_i, B_i);
  auto P3 = at::mm(A_r + A_i, B_r + B_i);
  auto AB_r = P1 - P2;
  auto AB_i = P3 - P1 - P2;

  // Complex multiplication: alpha * AB and beta * C
  // alpha * AB: (alpha_r + alpha_i*i) * (AB_r + AB_i*i) = (alpha_r*AB_r - alpha_i*AB_i) + (alpha_r*AB_i + alpha_i*AB_r)*i
  // beta * C: (beta_r + beta_i*i) * (C_r + C_i*i) = (beta_r*C_r - beta_i*C_i) + (beta_r*C_i + beta_i*C_r)*i
  auto out_real = at::view_as_real(out);
  out_real.select(-1, 0).copy_(
      (beta_r * C_r - beta_i * C_i) + (alpha_r * AB_r - alpha_i * AB_i));
  out_real.select(-1, 1).copy_(
      (beta_r * C_i + beta_i * C_r) + (alpha_r * AB_i + alpha_i * AB_r));
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
      "Complex baddbmm is using fallback implementation with real GEMM."
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto beta_c = beta.toComplexDouble();
  auto alpha_c = alpha.toComplexDouble();
  double beta_r = beta_c.real(), beta_i = beta_c.imag();
  double alpha_r = alpha_c.real(), alpha_i = alpha_c.imag();

  auto self_real = at::view_as_real(self);
  auto batch1_real = at::view_as_real(batch1);
  auto batch2_real = at::view_as_real(batch2);

  auto C_r = self_real.select(-1, 0);
  auto C_i = self_real.select(-1, 1);
  auto A_r = batch1_real.select(-1, 0);
  auto A_i = batch1_real.select(-1, 1);
  auto B_r = batch2_real.select(-1, 0);
  auto B_i = batch2_real.select(-1, 1);

  // Gauss-Strassen: 3 GEMMs for A @ B
  auto P1 = at::bmm(A_r, B_r);
  auto P2 = at::bmm(A_i, B_i);
  auto P3 = at::bmm(A_r + A_i, B_r + B_i);
  auto AB_r = P1 - P2;
  auto AB_i = P3 - P1 - P2;

  // Complex multiplication for alpha and beta
  auto out_real = at::view_as_real(out);
  out_real.select(-1, 0).copy_(
      (beta_r * C_r - beta_i * C_i) + (alpha_r * AB_r - alpha_i * AB_i));
  out_real.select(-1, 1).copy_(
      (beta_r * C_i + beta_i * C_r) + (alpha_r * AB_i + alpha_i * AB_r));
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

inline void dot_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(
      self.dim() == 1 && other.dim() == 1,
      "1D tensors expected, but got ",
      self.dim(),
      "D and ",
      other.dim(),
      "D tensors");
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "dot : expected both vectors to have same dtype, but found ",
      self.scalar_type(),
      " and ",
      other.scalar_type());
  TORCH_CHECK(
      self.numel() == other.numel(),
      "inconsistent tensor size, expected tensor [",
      self.numel(),
      "] and src [",
      other.numel(),
      "] to have the same number of elements, but got ",
      self.numel(),
      " and ",
      other.numel(),
      " elements respectively");
  TORCH_CHECK(
      (self.numel() <= INT_MAX) && (self.stride(0) <= INT_MAX) &&
          (other.stride(0) <= INT_MAX),
      "dot only supports n, incx, incy with the bound [val] <= %d",
      INT_MAX);
}

Tensor dot_xpu(const Tensor& self, const Tensor& other) {
  c10::DeviceGuard guard(self.device());

  if (self.is_complex()) {
    if (self.is_conj()) {
      if (other.is_conj()) {
        return (dot_xpu(self.conj(), other.conj())).conj();
       } else {
         return vdot_xpu(self.conj(), other);
       }
    } else if (other.is_conj()) {
      return vdot_xpu(other.conj(), self);
    }
  }

  dot_check(self, other);

  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  #if defined(USE_ONEMKL_XPU)
    return at::native::xpu::dot_xpu_mkl(self, other);
  #else
    return at::native::vdot(self.cpu(), other.cpu()).to(self.device());
  #endif
}

Tensor vdot_xpu(const Tensor& self, const Tensor& other) {
  c10::DeviceGuard guard(self.device());

  if (!self.is_complex()) {
    return dot_xpu(self, other);
  }

  if (self.is_conj()) {
    if (other.is_conj()) {
      return vdot_xpu(other.conj(), self.conj());
    } else {
      return dot_xpu(self.conj(), other);
    }
  } else if (other.is_conj()) {
    return (dot_xpu(self, other.conj())).conj();
  }

  dot_check(self, other);

  if (self._is_zerotensor() || other._is_zerotensor()) {
    return at::_efficientzerotensor({}, self.options());
  }

  #if defined(USE_ONEMKL_XPU)
    return at::native::xpu::vdot_xpu_mkl(self, other);
  #else
    return at::native::vdot(self.cpu(), other.cpu()).to(self.device());
  #endif
}

} // namespace at::native
