/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/GroupedBlas.h>
#include <ATen/native/xpu/sycltla/GroupMM.h>
#include <ATen/native/GroupedMMUtils.h>
#include <c10/util/env.h>

namespace at::native {

namespace {

bool is_grouped_mm_sycltla_available() {
#ifndef USE_SYCLTLA
  return false;
#else
  // Mirrors flash-attn style compile-time gating while allowing runtime opt-out.
  return c10::utils::check_env("PYTORCH_XPU_GROUPED_MM_SYCLTLA_ENABLED") != false;
#endif
}

bool force_grouped_mm_sycltla() {
  return c10::utils::check_env("PYTORCH_XPU_GROUPED_MM_FORCE_SYCLTLA") == true;
}

} // namespace

Tensor _grouped_mm_xpu(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<at::Tensor>& offs,
    const std::optional<at::Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {
  _grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  const auto out_dtype_ = _resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
    const bool bf16_fast_path =
      mat_a.dtype() == at::kBFloat16 && mat_b.dtype() == at::kBFloat16 &&
      out_dtype_ == at::kBFloat16;
  Tensor out = create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
  if (bf16_fast_path && is_grouped_mm_sycltla_available()) {
    at::xpu::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
  } else {
    TORCH_CHECK(
        !force_grouped_mm_sycltla(),
        "_grouped_mm_xpu: forced SYCLTLA path was requested by "
        "PYTORCH_XPU_GROUPED_MM_FORCE_SYCLTLA=1, but inputs/runtime are not "
        "eligible for bf16bf16_grouped_mm");
    _grouped_mm_fallback(mat_a, mat_b, offs, bias, out_dtype, out);
  }
  return out;
}

} // namespace at::native
