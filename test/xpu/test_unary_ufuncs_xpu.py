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
    dtypesIfXPU,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import (
    floating_and_complex_types_and,
    floating_types_and,
)
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_unary_ufuncs import TestUnaryUfuncs, TestUnaryUfuncsCUDADevice


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
# dtypesIfXPU (fp8 subnormal tests)
# ======================================================================
# Note: float16 fails on XPU: the XPU kernel converts -0.0 fp16 to fp8 as +0
# (0x00) while the CPU reference path (fp16 -> fp32 -> fp8) preserves the sign
# bit, giving -0 (0x80). Only 18 out of 1M elements are affected (all negative
# zeros at the fp16 precision boundary). float32 and bfloat16 work correctly.
# On CUDA, all three dtypes pass — the test was introduced as a regression test
# for a ptxas codegen bug on sm_100.

TestUnaryUfuncsCUDADevice.test_fp8_e4m3fn_conversion_subnormals = dtypesIfXPU(
    torch.float32, torch.bfloat16
)(TestUnaryUfuncsCUDADevice.test_fp8_e4m3fn_conversion_subnormals)

TestUnaryUfuncsCUDADevice.test_fp8_e5m2_conversion_subnormals = dtypesIfXPU(
    torch.float32, torch.bfloat16
)(TestUnaryUfuncsCUDADevice.test_fp8_e5m2_conversion_subnormals)

# ======================================================================
# Instantiate tests
# ======================================================================

instantiate_device_type_tests(
    TestUnaryUfuncs, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestUnaryUfuncsCUDADevice, globals(), only_for="xpu", allow_xpu=True
)


# Skip XPU unary ufunc tests that currently fail; tracked upstream.
# Each entry maps an exact generated test name to its tracking issue.
_xpu_skip_cases = {
    "TestUnaryUfuncsXPU": {
        "test_reference_numerics_extremal__refs_acos_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal__refs_acosh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal__refs_asin_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal__refs_asinh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal__refs_nn_functional_tanhshrink_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal__refs_tan_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal__refs_tanh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal_acosh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal_asin_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal_asinh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal_nn_functional_tanhshrink_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal_round_decimals_3_xpu_bfloat16": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal_tan_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_extremal_tanh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_large__refs_acosh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_large__refs_asinh_xpu_complex128": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_large__refs_asinh_xpu_complex32": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_large__refs_asinh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_large_acosh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_large_asinh_xpu_complex128": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_large_asinh_xpu_complex32": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_large_asinh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal__refs_asinh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal__refs_nn_functional_tanhshrink_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal_asinh_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal_nn_functional_tanhshrink_xpu_complex64": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal_polygamma_polygamma_n_1_xpu_float16": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal_polygamma_polygamma_n_2_xpu_float16": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal_polygamma_polygamma_n_3_xpu_float16": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal_polygamma_polygamma_n_4_xpu_float16": "https://github.com/intel/torch-xpu-ops/issues/2257",
        "test_reference_numerics_normal_round_decimals_3_xpu_bfloat16": "https://github.com/intel/torch-xpu-ops/issues/2257",
    },
}


def _apply_xpu_skips(_skip_cases):
    for _cls_name, _cases in _skip_cases.items():
        _cls = globals().get(_cls_name)
        if _cls is None:
            continue
        for _name, _issue in _cases.items():
            _method = getattr(_cls, _name, None)
            if _method is not None:
                setattr(
                    _cls,
                    _name,
                    unittest.skip(f"Skipped on XPU, see {_issue}")(_method),
                )


_apply_xpu_skips(_xpu_skip_cases)


if __name__ == "__main__":
    run_tests()
