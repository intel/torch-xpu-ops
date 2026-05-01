/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/ScaledGroupedMM.h>

#ifdef USE_SYCLTLA
#include <ATen/native/xpu/sycltla/ScaledGroupedMM.h>
#endif

namespace at::native::xpu {

bool is_scaled_grouped_mm_available() {
#ifdef USE_SYCLTLA
  return true;
#else
  return false;
#endif
}

void f8f8bf16_scaled_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    std::optional<at::Tensor> offs,
    at::Tensor& out) {
#ifdef USE_SYCLTLA
  at::xpu::detail::f8f8bf16_scaled_grouped_mm(
      mat_a, mat_b, scale_a, scale_b, offs, out);
#else
  TORCH_CHECK(
      false,
      "f8f8bf16_scaled_grouped_mm: torch-xpu-ops was not compiled with SYCLTLA support.");
#endif
}

} // namespace at::native::xpu
