/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/GroupedMM.h>

#ifdef USE_SYCLTLA
#include <ATen/native/xpu/sycltla/GroupedMM.h>
#endif

namespace at::native::xpu {

bool is_grouped_mm_available() {
#ifdef USE_SYCLTLA
  return true;
#else
  return false;
#endif
}

void bf16bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs,
    std::optional<at::Tensor> bias,
    at::Tensor& out) {
#ifdef USE_SYCLTLA
  at::xpu::detail::bf16bf16_grouped_mm(mat_a, mat_b, offs, bias, out);
#else
  TORCH_CHECK(
      false,
      "bf16bf16_grouped_mm: torch-xpu-ops was not compiled with SYCLTLA support.");
#endif
}

} // namespace at::native::xpu
