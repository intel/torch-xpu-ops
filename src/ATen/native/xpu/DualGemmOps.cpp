/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * Custom XPU operator registrations.
 *
 * Registers ops in the `xpu_ops` namespace so they can be called from Python
 * as torch.ops.xpu_ops.<name>.
 */

#include <ATen/native/transformers/xpu/dual_gemm/dual_gemm_api.h>
#include <torch/library.h>

namespace at::native::xpu {

// ---------------------------------------------------------------------------
// dual_gemm_silu_mul
//
// Fused kernel: out = silu(x @ w1.T) * (x @ w3.T)
// x  : [M, K] bfloat16 XPU tensor
// w1 : [N, K] bfloat16 XPU tensor
// w3 : [N, K] bfloat16 XPU tensor
// out: [M, N] bfloat16 XPU tensor
// ---------------------------------------------------------------------------
at::Tensor dual_gemm_silu_mul(
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w3) {
  return sycltla::dual_gemm_silu_mul(x, w1, w3);
}

// Abstract (meta) implementation: infers output shape/dtype without executing
// the real kernel.  Required for torch.compile / FakeTensor propagation.
at::Tensor dual_gemm_silu_mul_meta(
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w3) {
  // Output is [M, N] with the same dtype and device as x.
  const int64_t M = x.size(0);
  const int64_t N = w1.size(0);
  return at::empty({M, N}, x.options());
}

} // namespace at::native::xpu

// Register schema only (no default implementation).
TORCH_LIBRARY(xpu_ops, m) {
  m.def("dual_gemm_silu_mul(Tensor x, Tensor w1, Tensor w3) -> Tensor");
}

// Register the XPU kernel for the XPU dispatch key.
TORCH_LIBRARY_IMPL(xpu_ops, XPU, m) {
  m.impl("dual_gemm_silu_mul", TORCH_FN(at::native::xpu::dual_gemm_silu_mul));
}

// Register the abstract/meta implementation for FakeTensor / shape inference.
TORCH_LIBRARY_IMPL(xpu_ops, Meta, m) {
  m.impl(
      "dual_gemm_silu_mul", TORCH_FN(at::native::xpu::dual_gemm_silu_mul_meta));
}
