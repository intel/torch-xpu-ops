/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/core/Tensor.h>
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/linalg_qr_cpu_dispatch.h>
#include <torch/library.h>
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

void triangular_solve_kernel_xpu(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  TORCH_CHECK(
      A.scalar_type() == B.scalar_type(),
      "triangular_solve_kernel_xpu: A and B must have the same dtype");

#if defined(USE_ONEMKL_XPU)
  native::xpu::triangular_solve_mkl(
      A, B, left, upper, transpose, unitriangular);
#else
  triangular_solve_kernel_fallback(A, B, left, upper, transpose, unitriangular);
#endif // USE_ONEMKL_XPU
}

REGISTER_XPU_DISPATCH(triangular_solve_stub, &triangular_solve_kernel_xpu);

void geqrf_kernel_fallback(const Tensor& input, const Tensor& tau) {
  TORCH_WARN_ONCE(
      "torch.geqrf op is using CPU fallback implementation on XPU. "
      "Consider building with USE_ONEMKL_XPU=1 for better performance.");

  auto input_cpu = input.to(input.options().device(kCPU));
  auto tau_cpu = tau.to(tau.options().device(kCPU));
  geqrf_stub(at::kCPU, input_cpu, tau_cpu);
  input.copy_(input_cpu);
  tau.copy_(tau_cpu);
}

Tensor& orgqr_kernel_fallback(Tensor& result, const Tensor& tau) {
  TORCH_WARN_ONCE(
      "torch.linalg.householder_product/torch.orgqr op is using CPU fallback "
      "implementation on XPU. Consider building with USE_ONEMKL_XPU=1 for "
      "better performance.");

  auto result_cpu = result.to(result.options().device(kCPU));
  auto tau_cpu = tau.to(tau.options().device(kCPU));
  orgqr_stub(at::kCPU, result_cpu, tau_cpu);
  result.copy_(result_cpu);
  return result;
}

void geqrf_kernel_xpu(const Tensor& input, const Tensor& tau) {
#if defined(USE_ONEMKL_XPU)
  native::xpu::geqrf_mkl(input, tau);
#else
  geqrf_kernel_fallback(input, tau);
#endif // USE_ONEMKL_XPU
}

REGISTER_XPU_DISPATCH(geqrf_stub, &geqrf_kernel_xpu);

Tensor& orgqr_kernel_xpu(Tensor& result, const Tensor& tau) {
#if defined(USE_ONEMKL_XPU)
  if (result.is_complex()) {
    return native::xpu::ungqr_mkl(result, tau);
  } else {
    return native::xpu::orgqr_mkl(result, tau);
  }
#else
  return orgqr_kernel_fallback(result, tau);
#endif // USE_ONEMKL_XPU
}

REGISTER_XPU_DISPATCH(orgqr_stub, &orgqr_kernel_xpu);

std::tuple<Tensor, Tensor> geqrf_xpu(const Tensor& input) {
  TORCH_CHECK(
      input.dim() >= 2, "torch.geqrf: input must have at least 2 dimensions.");

  Tensor qr = cloneBatchedColumnMajor(input);

  auto tau_shape = input.sizes().vec();
  tau_shape.pop_back();
  tau_shape.back() = std::min(input.size(-2), input.size(-1));
  Tensor tau = input.new_empty(tau_shape);

  geqrf_stub(input.device().type(), qr, tau);
  return std::make_tuple(std::move(qr), std::move(tau));
}

Tensor& linalg_householder_product_out_xpu(
    const Tensor& input,
    const Tensor& tau,
    Tensor& result) {
  TORCH_CHECK(
      input.dim() >= 2,
      "torch.linalg.householder_product: input must have at least 2 dimensions.");
  TORCH_CHECK(
      input.size(-2) >= input.size(-1),
      "torch.linalg.householder_product: input.shape[-2] must be greater than or equal to input.shape[-1]");
  TORCH_CHECK(
      input.size(-1) >= tau.size(-1),
      "torch.linalg.householder_product: input.shape[-1] must be greater than or equal to tau.shape[-1]");
  TORCH_CHECK(
      input.dim() - tau.dim() == 1,
      "torch.linalg.householder_product: Expected tau to have one dimension less than input, but got tau.ndim equal to ",
      tau.dim(),
      " and input.ndim is equal to ",
      input.dim());
  if (input.dim() > 2) {
    auto expected_batch_tau_shape =
        IntArrayRef(input.sizes().data(), input.dim() - 2);
    auto actual_batch_tau_shape =
        IntArrayRef(tau.sizes().data(), tau.dim() - 1);
    TORCH_CHECK(
        actual_batch_tau_shape.equals(expected_batch_tau_shape),
        "torch.linalg.householder_product: Expected batch dimensions of tau to be equal to input.shape[:-2], but got ",
        actual_batch_tau_shape);
  }
  TORCH_CHECK(
      input.scalar_type() == tau.scalar_type(),
      "torch.linalg.householder_product: tau dtype ",
      tau.scalar_type(),
      " does not match input dtype ",
      input.scalar_type());
  checkSameDevice("torch.linalg.householder_product", tau, input, "tau");
  checkSameDevice("torch.linalg.householder_product", result, input);
  checkLinalgCompatibleDtype("torch.linalg.householder_product", result, input);

  bool result_input_same_type = (result.scalar_type() == input.scalar_type());
  bool result_equal_expected_shape = result.sizes().equals(input.sizes());
  bool is_batched_column_major = false;
  if (result.dim() >= 2) {
    is_batched_column_major = result.mT().is_contiguous();
  }

  bool copy_needed = (result.numel() != 0 && !is_batched_column_major);
  copy_needed |= !result_input_same_type;
  copy_needed |= (result.numel() != 0 && !result_equal_expected_shape);

  auto householder_product_out_helper =
      [](const Tensor& in, const Tensor& t, Tensor& out) -> Tensor& {
    if (out.numel() == 0) {
      out.resize_as_(in.mT(), MemoryFormat::Contiguous);
      out.transpose_(-2, -1);
    }

    TORCH_INTERNAL_ASSERT(out.mT().is_contiguous());
    TORCH_INTERNAL_ASSERT(out.sizes().equals(in.sizes()));

    Tensor tau_contig = t;
    if (!t.is_contiguous()) {
      tau_contig = at::empty(t.sizes(), t.options(), MemoryFormat::Contiguous);
      tau_contig.copy_(t);
    }

    out.copy_(in);
    out = orgqr_stub(out.device().type(), out, tau_contig);
    return out;
  };

  if (copy_needed) {
    Tensor result_tmp = at::empty({0}, input.options());
    result_tmp = householder_product_out_helper(input, tau, result_tmp);
    at::native::resize_output(result, result_tmp.sizes());
    result.copy_(result_tmp);
  } else {
    result = householder_product_out_helper(input, tau, result);
  }

  return result;
}

Tensor linalg_householder_product_xpu(const Tensor& input, const Tensor& tau) {
  Tensor result = at::empty({0}, input.options());
  linalg_householder_product_out_xpu(input, tau, result);
  return result;
}
} // namespace at::native

TORCH_LIBRARY_IMPL(aten, XPU, m) {
  m.impl("geqrf", TORCH_FN(at::native::geqrf_xpu));
  m.impl(
      "linalg_householder_product",
      TORCH_FN(at::native::linalg_householder_product_xpu));
  m.impl(
      "linalg_householder_product.out",
      TORCH_FN(at::native::linalg_householder_product_out_xpu));
}
