/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/ATen.h>

namespace sycltla {

// Returns true if the sycltla dual_gemm kernel is compiled in.
bool is_dual_gemm_available();

/**
 * Fused dual GEMM with SiLU-Mul epilogue.
 *
 * Computes:  out = silu(x @ w1.T) * (x @ w3.T)
 *
 * This fuses two GEMM operations that share the same A matrix (x) into a
 * single kernel, loading x from HBM only once.
 *
 * @param x      Input tensor of shape [M, K], contiguous, bf16 or fp16.
 * @param w1     Weight matrix of shape [N, K], contiguous, same dtype as x.
 * @param w3     Weight matrix of shape [N, K], contiguous, same dtype as x.
 * @return       Output tensor of shape [M, N], same dtype as x.
 */
at::Tensor dual_gemm_silu_mul(
    const at::Tensor& x,
    const at::Tensor& w1,
    const at::Tensor& w3);

} // namespace sycltla
