/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Tensor.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/ComplexHelper.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebraUtils.h>
#if defined(USE_ONEMKL_XPU)
#include <ATen/native/xpu/mkl/BatchLinearAlgebra.h>
#endif // USE_ONEMKL_XPU

namespace at::native {

void lu_solve_kernel_xpu(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType trans) {
#if defined(USE_ONEMKL_XPU)
  native::xpu::lu_solve_mkl(LU, pivots, B, trans);
#else
  const auto LU_cpu = LU.to(LU.options().device(kCPU));
  const auto pivots_cpu = pivots.to(pivots.options().device(kCPU));
  auto B_cpu = B.to(B.options().device(kCPU));

  lu_solve_stub(at::kCPU, LU_cpu, pivots_cpu, B_cpu, trans);

  B.copy_(B_cpu);
#endif // USE_ONEMKL_XPU
}

REGISTER_XPU_DISPATCH(lu_solve_stub, &lu_solve_kernel_xpu);

void lu_factor_kernel_fallback(
    const Tensor& input,
    const Tensor& pivots,
    const Tensor& infos,
    bool compute_pivots) {
  auto input_cpu = input.to(input.options().device(kCPU));
  auto pivots_cpu = pivots.to(pivots.options().device(kCPU));
  const auto infos_cpu = infos.to(infos.options().device(kCPU));

  lu_factor_stub(at::kCPU, input_cpu, pivots_cpu, infos_cpu, compute_pivots);

  input.copy_(input_cpu);
  pivots.copy_(pivots_cpu);
  infos.copy_(infos_cpu);
}

void lu_factor_kernel_xpu(
    const Tensor& input,
    const Tensor& pivots,
    const Tensor& infos,
    bool compute_pivots) {
#if defined(USE_ONEMKL_XPU)
  int64_t batch_size = native::batchCount(input);
  // TODO: optimize lu_factor performance on XPU when batch_size = 1
  if (batch_size == 1) {
    lu_factor_kernel_fallback(input, pivots, infos, compute_pivots);
  } else {
    native::xpu::lu_factor_mkl(input, pivots, infos, compute_pivots);
  }
#else
  lu_factor_kernel_fallback(input, pivots, infos, compute_pivots);
#endif // USE_ONEMKL_XPU
}

REGISTER_XPU_DISPATCH(lu_factor_stub, &lu_factor_kernel_xpu);

at::Tensor copy_to_cpu_preserving_strides_and_conj(const Tensor& xpu_tensor) {
  if (xpu_tensor.is_complex()) {
    auto cpu_tensor = at::empty_strided(
        xpu_tensor.sizes(),
        xpu_tensor.strides(),
        xpu_tensor.options().device(kCPU));
    cpu_tensor._set_conj(xpu_tensor.is_conj());
    cpu_tensor.copy_(xpu_tensor);

    return cpu_tensor;
  } else {
    return xpu_tensor.to(xpu_tensor.options().device(kCPU));
  }
}

void triangular_solve_kernel_fallback(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  TORCH_WARN_ONCE(
      "torch.linalg.solve_triangular op is using fallback implementation. "
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  // triangular_solve_stub sets TransposeType based on A and B tensors and its
  // conjugation, copying to CPU solves conjugation leading to improper
  // TransposeType. So we need to preserve the conjugation and strides when
  // copying to CPU.
  auto A_cpu = copy_to_cpu_preserving_strides_and_conj(A);
  auto B_cpu = copy_to_cpu_preserving_strides_and_conj(B);

  triangular_solve_stub(
      DeviceType::CPU, A_cpu, B_cpu, left, upper, transpose, unitriangular);

  B.copy_(B_cpu);
}

void triangular_solve_kernel_mkl(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
#if defined(USE_ONEMKL_XPU)
  native::xpu::triangular_solve_mkl(
      A, B, left, upper, transpose, unitriangular);
#else
  triangular_solve_kernel_fallback(A, B, left, upper, transpose, unitriangular);
#endif // USE_ONEMKL_XPU
}

REGISTER_XPU_DISPATCH(triangular_solve_stub, &triangular_solve_kernel_mkl);

} // namespace at::native
