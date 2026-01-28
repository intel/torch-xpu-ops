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
#endif
#include <torch/library.h>

namespace at::native {

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
  // CPU fallback for complex matmul when oneMKL is not available
  TORCH_WARN_ONCE(
      "Complex matmul on XPU is falling back to CPU. ",
      "Compile with USE_ONEMKL_XPU=1 for native XPU support.");
  auto self_cpu = self.to(at::kCPU);
  auto mat2_cpu = mat2.to(at::kCPU);
  auto out_cpu = at::mm(self_cpu, mat2_cpu);
  out.copy_(out_cpu);
  return out;
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
  // CPU fallback for complex bmm when oneMKL is not available
  TORCH_WARN_ONCE(
      "Complex bmm on XPU is falling back to CPU. ",
      "Compile with USE_ONEMKL_XPU=1 for native XPU support.");
  auto self_cpu = self.to(at::kCPU);
  auto mat2_cpu = mat2.to(at::kCPU);
  auto out_cpu = at::bmm(self_cpu, mat2_cpu);
  out.copy_(out_cpu);
  return out;
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
  // CPU fallback for complex addmm when oneMKL is not available
  TORCH_WARN_ONCE(
      "Complex addmm on XPU is falling back to CPU. ",
      "Compile with USE_ONEMKL_XPU=1 for native XPU support.");
  auto self_cpu = self.to(at::kCPU);
  auto mat1_cpu = mat1.to(at::kCPU);
  auto mat2_cpu = mat2.to(at::kCPU);
  auto out_cpu = at::addmm(self_cpu, mat1_cpu, mat2_cpu, beta, alpha);
  out.copy_(out_cpu);
  return out;
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
  // CPU fallback for complex baddbmm when oneMKL is not available
  TORCH_WARN_ONCE(
      "Complex baddbmm on XPU is falling back to CPU. ",
      "Compile with USE_ONEMKL_XPU=1 for native XPU support.");
  auto self_cpu = self.to(at::kCPU);
  auto batch1_cpu = batch1.to(at::kCPU);
  auto batch2_cpu = batch2.to(at::kCPU);
  auto out_cpu = at::baddbmm(self_cpu, batch1_cpu, batch2_cpu, beta, alpha);
  out.copy_(out_cpu);
  return out;
#endif // USE_ONEMKL_XPU
}

} // namespace at::native
