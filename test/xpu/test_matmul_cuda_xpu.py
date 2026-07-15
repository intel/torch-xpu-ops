# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: intel"]

import unittest

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyOn,
    skipXPU,
    tol as xtol,
    toleranceOverride,
)
from torch.testing._internal.common_utils import (
    IS_JETSON,
    MI200_ARCH,
    parametrize,
    run_tests,
    runOnRocmArch,
    TestCase,
)

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_matmul_cuda import TestMatmulCuda, TestMixedDtypesLinearCuda


# ======================================================================
# Override CUDA-only methods to also run on XPU
# ======================================================================

# NOTE: we added the `device` parameter here - needs to be also applied
# when porting to upstream


@onlyOn(["cuda", "xpu"])
def _test_cublas_and_lt_reduced_precision_fp16_accumulate(self, device):
    orig_fp16_accumulate = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
    x = torch.rand(32, 512, 512, device=device, dtype=torch.half)
    w = torch.rand(512, 512, device=device, dtype=torch.half)
    b = torch.rand(512, device=device, dtype=torch.half)
    out = torch.nn.functional.linear(x, w, b)
    out_cpu = torch.nn.functional.linear(x.cpu(), w.cpu(), b.cpu())
    self.assertEqual(out, out_cpu, atol=5e-3, rtol=8e-3)

    a = torch.rand(16, 128, 128, device=device, dtype=torch.half)
    b = torch.rand(16, 128, 128, device=device, dtype=torch.half)
    c = torch.rand(16, 128, 128, device=device, dtype=torch.half)
    out = torch.baddbmm(a, b, c)
    out_cpu = torch.baddbmm(a.cpu(), b.cpu(), c.cpu())
    self.assertEqual(out, out_cpu, atol=1e-3, rtol=5e-3)

    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accumulate


TestMatmulCuda.test_cublas_and_lt_reduced_precision_fp16_accumulate = (
    _test_cublas_and_lt_reduced_precision_fp16_accumulate
)


@onlyOn(["cuda", "xpu"])
@toleranceOverride({torch.float16: xtol(atol=1e-3, rtol=3e-3)})
@dtypes(torch.float16)
def _test_cublas_addmm_alignment(self, device, dtype):
    for idx in range(3):
        for offset in range(1, 3):
            offsets = [0, 0, 0]
            offsets[idx] = offset
            x_offset, a_offset, b_offset = offsets
            A = torch.rand(
                (5120 * 2560 + a_offset),
                requires_grad=True,
                dtype=dtype,
                device=device,
            )
            A = A[a_offset:].reshape(5120, 2560)
            X = torch.rand(
                (26 * 2560 + x_offset),
                requires_grad=True,
                dtype=dtype,
                device=device,
            )
            X = X[x_offset:].reshape(26, 1, 2560)
            B = torch.rand(
                (5120 + b_offset),
                requires_grad=True,
                dtype=dtype,
                device=device,
            )
            B = B[b_offset:].reshape(5120)
            out = torch.nn.functional.linear(X, A, B)
            self.assertEqual(out, torch.matmul(X, A.transpose(1, 0)) + B)


TestMatmulCuda.test_cublas_addmm_alignment = _test_cublas_addmm_alignment


@onlyOn(["cuda", "xpu"])
@unittest.skipIf(IS_JETSON, "Too large for Jetson")
@toleranceOverride({torch.float32: xtol(atol=1e-5, rtol=1.1e-5)})
@dtypes(torch.float32, torch.float16, torch.bfloat16)
@parametrize(
    "batch_size, N, M, P",
    [
        (2, 100, 100, 100),
        (2, 1000, 1000, 1000),
        (1, 10000, 1000, 10000),
        (1, 10000, 10000, 10000),
    ],
    name_fn=lambda batch_size, N, M, P: f"{batch_size}_{N}_{M}_{P}",
)
def _test_cublas_baddbmm_large_input(self, device, batch_size, N, M, P, dtype):
    cpu_dtype = dtype
    if dtype == torch.float16 or dtype == torch.bfloat16:
        cpu_dtype = torch.float32

    M1 = torch.rand((N, M), device=device, dtype=dtype)
    M2 = torch.rand((M, P), device=device, dtype=dtype)
    A = torch.rand((N, P), device=device, dtype=dtype)

    def _convert_to_cpu(t):
        return t.to(device="cpu", dtype=cpu_dtype)

    M1_cpu, M2_cpu, A_cpu = map(_convert_to_cpu, [M1, M2, A])

    out1_cpu = torch.nn.functional.linear(M1_cpu, M2_cpu.t(), A_cpu).to(dtype=dtype)
    out1_gpu = torch.nn.functional.linear(M1, M2.t(), A).cpu()
    self.assertEqual(out1_cpu, out1_gpu)

    if N == M and M == P:
        M2_eye = torch.eye(N, device=device, dtype=dtype)
        out1_eye_gpu = torch.nn.functional.linear(M1, M2_eye.t(), torch.zeros_like(A))
        if runOnRocmArch(MI200_ARCH) and dtype == torch.float16:
            self.assertEqual(
                M1_cpu.to(dtype=dtype), out1_eye_gpu.cpu(), atol=1e-4, rtol=0.001
            )
        else:
            self.assertEqual(M1_cpu.to(dtype=dtype), out1_eye_gpu.cpu())

    def _expand_to_batch(t: torch.Tensor):
        return t.expand((batch_size,) + t.size())

    alpha, beta = 1.0, 1.0
    M1, M2, A, M1_cpu, M2_cpu, A_cpu = map(
        _expand_to_batch, [M1, M2, A, M1_cpu, M2_cpu, A_cpu]
    )

    out2_cpu = torch.baddbmm(A_cpu, M1_cpu, M2_cpu, beta=beta, alpha=alpha).to(
        dtype=dtype
    )
    out2_gpu = torch.baddbmm(A, M1, M2, beta=beta, alpha=alpha).cpu()
    self.assertEqual(out2_cpu, out2_gpu)

    if N == M and M == P:
        M2_eye = torch.eye(N, device=device, dtype=dtype).expand(batch_size, N, N)
        out2_eye_gpu = torch.baddbmm(
            torch.zeros_like(A), M1, M2_eye, beta=beta, alpha=alpha
        )
        if runOnRocmArch(MI200_ARCH) and dtype == torch.float16:
            self.assertEqual(
                M1_cpu.to(dtype=dtype), out2_eye_gpu.cpu(), atol=1e-4, rtol=0.001
            )
        else:
            self.assertEqual(M1_cpu.to(dtype=dtype), out2_eye_gpu.cpu())

    self.assertEqual(out1_gpu, out2_gpu[0])


TestMatmulCuda.test_cublas_baddbmm_large_input = _test_cublas_baddbmm_large_input


# ======================================================================
# Keep CUDA/ROCm-specific coverage, but skip these test on XPU
# ======================================================================

TestMatmulCuda.test_grouped_gemm_rocm_ck_flag = skipXPU(
    TestMatmulCuda.test_grouped_gemm_rocm_ck_flag
)
TestMatmulCuda.test_grouped_gemm_doubly_non_contiguous = skipXPU(
    TestMatmulCuda.test_grouped_gemm_doubly_non_contiguous
)


# ======================================================================
# Instantiate tests for XPU
# ======================================================================

instantiate_device_type_tests(TestMatmulCuda, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(
    TestMixedDtypesLinearCuda, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
