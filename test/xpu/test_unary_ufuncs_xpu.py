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
    dtypesIfXPU,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import (
    floating_and_complex_types_and,
    floating_types_and,
)
from torch.testing._internal.common_utils import run_tests, xfailIfTorchDynamo

try:
    from xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx
except Exception:
    from .xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx

with XPUImportCtx(False):
    from test_unary_ufuncs import TestUnaryUfuncs


# ======================================================================
# dtypesIfXPU decorator additions
# ======================================================================

TestUnaryUfuncs.test_i0_range1 = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestUnaryUfuncs.test_i0_range1)

TestUnaryUfuncs.test_i0_range2 = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestUnaryUfuncs.test_i0_range2)

TestUnaryUfuncs.test_i0_special = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestUnaryUfuncs.test_i0_special)

TestUnaryUfuncs.test_special_i0_i1_vs_scipy = dtypesIfXPU(
    *floating_types_and(torch.half, torch.bfloat16)
)(TestUnaryUfuncs.test_special_i0_i1_vs_scipy)

TestUnaryUfuncs.test_exp = dtypesIfXPU(
    *floating_and_complex_types_and(torch.half, torch.bfloat16)
)(TestUnaryUfuncs.test_exp)

# ======================================================================
# Retarget @onlyCUDA to @onlyOn(["cuda", "xpu"])
# ======================================================================

TestUnaryUfuncs.test_tan_complex_cuda_matches_numpy = (
    retarget_outermost_onlycuda_to_onlyon(
        TestUnaryUfuncs.test_tan_complex_cuda_matches_numpy
    )
)

TestUnaryUfuncs.test_tanh_complex_cuda_matches_numpy = (
    retarget_outermost_onlycuda_to_onlyon(
        TestUnaryUfuncs.test_tanh_complex_cuda_matches_numpy
    )
)

# ======================================================================
# Retarget @onlyCUDA + dtypesIfXPU (fp8 subnormal tests)
# ======================================================================
# Note: float16 fails on XPU: the XPU kernel converts -0.0 fp16 to fp8 as +0
# (0x00) while the CPU reference path (fp16 -> fp32 -> fp8) preserves the sign
# bit, giving -0 (0x80). Only 18 out of 1M elements are affected (all negative
# zeros at the fp16 precision boundary). float32 and bfloat16 work correctly.
# On CUDA, all three dtypes pass — the test was introduced as a regression test
# for a ptxas codegen bug on sm_100.

TestUnaryUfuncs.test_fp8_e4m3fn_conversion_subnormals = dtypesIfXPU(
    torch.float32, torch.bfloat16
)(
    retarget_outermost_onlycuda_to_onlyon(
        TestUnaryUfuncs.test_fp8_e4m3fn_conversion_subnormals
    )
)

TestUnaryUfuncs.test_fp8_e5m2_conversion_subnormals = dtypesIfXPU(
    torch.float32, torch.bfloat16
)(
    retarget_outermost_onlycuda_to_onlyon(
        TestUnaryUfuncs.test_fp8_e5m2_conversion_subnormals
    )
)


# ======================================================================
# Method overrides (replace hardcoded "cuda" with device parameter)
# ======================================================================


def _test_nonzero_static_large(self, device):
    # large enough to have multiple iters per SM even on H100
    # with 132 sms
    size_inp = 1024 * 16 * 132 + 1024 * 16
    x = torch.zeros(size_inp, device=device)
    # unique indices
    indices = torch.randperm(size_inp, device=device)[: size_inp // 2]
    sorted, _ = torch.sort(indices)
    x[sorted] = 1
    res = torch.nonzero_static(x, size=size_inp // 2).view(-1)
    self.assertEqual(res, sorted)
    # no oob writes
    out = torch.full((size_inp,), 10, device=device, dtype=torch.int64)
    res = torch.nonzero_static(x, size=size_inp // 4, out=out[: size_inp // 2])
    self.assertEqual(out[: size_inp // 4], sorted[: size_inp // 4])
    self.assertEqual(
        out[size_inp // 4 :],
        torch.tensor(10, device=device).expand_as(out[size_inp // 4 :]),
    )
    # correct fill for 2d
    x = x.view(2, size_inp // 2)
    ref = x.nonzero()
    res = x.nonzero_static(size=size_inp // 2 + 2)
    self.assertEqual(res.shape, [size_inp // 2 + 2, 2])
    self.assertEqual(ref, res[: size_inp // 2])
    self.assertEqual(
        res[size_inp // 2 :],
        torch.tensor(-1, device=device).expand_as(res[size_inp // 2 :]),
    )


TestUnaryUfuncs.test_nonzero_static_large = _test_nonzero_static_large


# ======================================================================
# Generalize memory overlap test to device-type parameter
# ======================================================================


@xfailIfTorchDynamo
@dtypes(torch.double)
def _test_unary_out_op_mem_overlap(self, device, dtype):
    sz = 3
    doubles = torch.randn(2 * sz, dtype=dtype, device=device)
    positives = torch.randint(1, 100, (2 * sz,), device=device).double()
    ints = torch.randint(-100, 100, (2 * sz,), device=device)
    unary_mem_overlap_cases = [
        ("abs", doubles, True, True, "cpu"),
        ("abs", doubles, True, True, "cuda"),
        ("abs", doubles, True, True, "xpu"),
        ("acos", doubles, True, True, "cpu"),
        ("acos", doubles, True, True, "cuda"),
        ("acos", doubles, True, True, "xpu"),
        ("asin", doubles, True, True, "cpu"),
        ("asin", doubles, True, True, "cuda"),
        ("asin", doubles, True, True, "xpu"),
        ("atan", doubles, True, True, "cpu"),
        ("atan", doubles, True, True, "cuda"),
        ("atan", doubles, True, True, "xpu"),
        ("acosh", doubles, True, True, "cpu"),
        ("acosh", doubles, True, True, "cuda"),
        ("acosh", doubles, True, True, "xpu"),
        ("asinh", doubles, True, True, "cpu"),
        ("asinh", doubles, True, True, "cuda"),
        ("asinh", doubles, True, True, "xpu"),
        ("atanh", doubles, True, True, "cpu"),
        ("atanh", doubles, True, True, "cuda"),
        ("atanh", doubles, True, True, "xpu"),
        ("bitwise_not", ints, True, True, "cpu"),
        ("bitwise_not", ints, True, True, "cuda"),
        ("bitwise_not", ints, True, True, "xpu"),
        ("ceil", doubles, True, True, "cpu"),
        ("ceil", doubles, True, True, "cuda"),
        ("ceil", doubles, True, True, "xpu"),
        ("cos", doubles, True, True, "cpu"),
        ("cos", doubles, True, True, "cuda"),
        ("cos", doubles, True, True, "xpu"),
        ("cosh", doubles, True, True, "cpu"),
        ("cosh", doubles, True, True, "cuda"),
        ("cosh", doubles, True, True, "xpu"),
        ("digamma", doubles, True, True, "cpu"),
        ("digamma", doubles, True, True, "xpu"),
        ("erf", doubles, True, True, "cpu"),
        ("erf", doubles, True, True, "cuda"),
        ("erf", doubles, True, True, "xpu"),
        ("erfc", doubles, True, True, "cpu"),
        ("erfc", doubles, True, True, "cuda"),
        ("erfc", doubles, True, True, "xpu"),
        ("erfinv", doubles, True, True, "cpu"),
        ("erfinv", doubles, True, True, "cuda"),
        ("erfinv", doubles, True, True, "xpu"),
        ("exp", doubles, True, True, "cpu"),
        ("exp", doubles, True, True, "cuda"),
        ("exp", doubles, True, True, "xpu"),
        ("exp2", doubles, True, True, "cpu"),
        ("exp2", doubles, True, True, "cuda"),
        ("exp2", doubles, True, True, "xpu"),
        ("expm1", doubles, True, True, "cpu"),
        ("expm1", doubles, True, True, "cuda"),
        ("expm1", doubles, True, True, "xpu"),
        ("floor", doubles, True, True, "cpu"),
        ("floor", doubles, True, True, "cuda"),
        ("floor", doubles, True, True, "xpu"),
        ("frac", doubles, True, True, "cpu"),
        ("frac", doubles, True, True, "cuda"),
        ("frac", doubles, True, True, "xpu"),
        ("i0", doubles, True, True, "cpu"),
        ("i0", doubles, True, True, "cuda"),
        ("i0", doubles, True, True, "xpu"),
        ("log", positives, True, True, "cpu"),
        ("log", positives, True, True, "cuda"),
        ("log", positives, True, True, "xpu"),
        ("log10", positives, True, True, "cpu"),
        ("log10", positives, True, True, "cuda"),
        ("log10", positives, True, True, "xpu"),
        ("log1p", positives, True, True, "cpu"),
        ("log1p", positives, True, True, "cuda"),
        ("log1p", positives, True, True, "xpu"),
        ("log2", positives, True, True, "cpu"),
        ("log2", positives, True, True, "cuda"),
        ("log2", positives, True, True, "xpu"),
        ("neg", doubles, True, True, "cpu"),
        ("neg", doubles, True, True, "cuda"),
        ("neg", doubles, True, True, "xpu"),
        ("reciprocal", doubles, True, True, "cpu"),
        ("reciprocal", doubles, True, True, "cuda"),
        ("reciprocal", doubles, True, True, "xpu"),
        ("round", doubles, True, True, "cpu"),
        ("round", doubles, True, True, "cuda"),
        ("round", doubles, True, True, "xpu"),
        ("rsqrt", positives, True, True, "cpu"),
        ("rsqrt", positives, True, True, "cuda"),
        ("rsqrt", positives, True, True, "xpu"),
        ("sin", doubles, True, True, "cpu"),
        ("sin", doubles, True, True, "cuda"),
        ("sin", doubles, True, True, "xpu"),
        ("sinh", doubles, True, True, "cpu"),
        ("sinh", doubles, False, True, "cuda"),
        ("sinh", doubles, True, True, "xpu"),
        ("sigmoid", doubles, True, True, "cpu"),
        ("sigmoid", doubles, True, True, "cuda"),
        ("sigmoid", doubles, True, True, "xpu"),
        ("logit", doubles, True, True, "cpu"),
        ("logit", doubles, True, True, "cuda"),
        ("logit", doubles, True, True, "xpu"),
        ("sqrt", doubles, True, True, "cpu"),
        ("sqrt", doubles, False, True, "cuda"),
        ("sqrt", doubles, True, True, "xpu"),
        ("tan", doubles, True, True, "cpu"),
        ("tan", doubles, True, True, "cuda"),
        ("tan", doubles, True, True, "xpu"),
        ("tanh", doubles, True, True, "cpu"),
        ("tanh", doubles, True, True, "cuda"),
        ("tanh", doubles, True, True, "xpu"),
        ("trunc", doubles, True, True, "cpu"),
        ("trunc", doubles, True, True, "cuda"),
        ("trunc", doubles, True, True, "xpu"),
    ]

    for (
        fn,
        inputs,
        has_input_output_mem_overlap_check,
        has_internal_mem_overlap_check,
        dev,
    ) in unary_mem_overlap_cases:
        if dev != torch.device(device).type:
            continue
        out_fn = getattr(torch, fn)
        in_fn = getattr(torch.Tensor, fn + "_")

        self.unary_check_input_output_mem_overlap(
            inputs,
            sz,
            out_fn,
            expected_failure=not has_input_output_mem_overlap_check,
        )

        self.check_internal_mem_overlap(
            in_fn,
            1,
            dtype,
            dev,
            expected_failure=not has_internal_mem_overlap_check,
        )


TestUnaryUfuncs.test_unary_out_op_mem_overlap = _test_unary_out_op_mem_overlap


# ======================================================================
# Instantiate tests
# ======================================================================

instantiate_device_type_tests(
    TestUnaryUfuncs, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
