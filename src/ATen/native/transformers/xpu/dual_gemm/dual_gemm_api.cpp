/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/transformers/xpu/dual_gemm/dual_gemm_api.h>

#ifdef USE_SYCLTLA
#include <ATen/xpu/XPUContext.h>

namespace sycltla {
// Forward declaration of the sycltla implementation
// (dual_gemm/sycltla/dual_gemm.cpp)
at::Tensor dual_gemm_silu_mul_impl(
    sycl::queue& queue,
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w3);
} // namespace sycltla
#endif // USE_SYCLTLA

namespace sycltla {

bool is_dual_gemm_available() {
#ifndef USE_SYCLTLA
  return false;
#else
  return true;
#endif
}

at::Tensor dual_gemm_silu_mul(
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w3) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false,
      "dual_gemm_silu_mul: Torch XPU was not compiled with SYCLTLA support.");
  return at::Tensor();
#else
  TORCH_CHECK(x.is_xpu(), "dual_gemm_silu_mul: x must be on XPU");
  TORCH_CHECK(w1.is_xpu(), "dual_gemm_silu_mul: w1 must be on XPU");
  TORCH_CHECK(w3.is_xpu(), "dual_gemm_silu_mul: w3 must be on XPU");

  TORCH_CHECK(
      x.dtype() == at::kBFloat16,
      "dual_gemm_silu_mul: only bfloat16 is currently supported, got ",
      x.dtype());
  TORCH_CHECK(
      w1.dtype() == at::kBFloat16 && w3.dtype() == at::kBFloat16,
      "dual_gemm_silu_mul: w1 and w3 must be bfloat16");

  TORCH_CHECK(x.dim() == 2, "dual_gemm_silu_mul: x must be 2-D [M, K]");
  TORCH_CHECK(w1.dim() == 2, "dual_gemm_silu_mul: w1 must be 2-D [N, K]");
  TORCH_CHECK(w3.dim() == 2, "dual_gemm_silu_mul: w3 must be 2-D [N, K]");

  TORCH_CHECK(x.is_contiguous(), "dual_gemm_silu_mul: x must be contiguous");
  TORCH_CHECK(w1.is_contiguous(), "dual_gemm_silu_mul: w1 must be contiguous");
  TORCH_CHECK(w3.is_contiguous(), "dual_gemm_silu_mul: w3 must be contiguous");

  const int64_t K = x.size(1);
  TORCH_CHECK(
      w1.size(1) == K && w3.size(1) == K,
      "dual_gemm_silu_mul: x, w1, w3 must share the same K dimension");
  TORCH_CHECK(
      w1.size(0) == w3.size(0),
      "dual_gemm_silu_mul: w1 and w3 must have the same N dimension");

  // The sycl-tla DualGemm kernel expects B matrices in [K, N] row-major
  // (i.e. already transposed relative to the [N, K] weight layout).
  // Transpose w1 and w3 so the kernel computes x @ w1.T correctly.
  at::Tensor w1_t = w1.t().contiguous();
  at::Tensor w3_t = w3.t().contiguous();

  auto sycl_queue = at::xpu::getCurrentXPUStream().queue();
  return sycltla::dual_gemm_silu_mul_impl(sycl_queue, x, w1_t, w3_t);
#endif
}

} // namespace sycltla
