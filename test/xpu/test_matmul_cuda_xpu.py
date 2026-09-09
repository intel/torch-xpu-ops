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


import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyOn,
    skipXPU,
    tol as xtol,
    toleranceOverride,
)
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx
except Exception:
    from .xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx

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


# ======================================================================
# Only replace the onlyCUDA decorator with onlyOn(["cuda", "xpu"])
# ======================================================================

TestMatmulCuda.test_cublas_baddbmm_large_input = retarget_outermost_onlycuda_to_onlyon(
    TestMatmulCuda.test_cublas_baddbmm_large_input
)


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
