# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression test for the XPU _assert_async kernel across dispatched dtypes.

A firing SYCL_KERNEL_ASSERT aborts the process, so the zero-input cases run in
a subprocess and check the return code and the assert text.
"""

import subprocess
import sys

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

DTYPES = [
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bool",
]
FP64_DTYPES = ("float64", "complex128")

SCRIPT = """
import torch
torch._assert_async(torch.tensor(0, dtype=torch.{dtype}, device="xpu"))
torch.xpu.synchronize()
"""


class TestAssertAsync(TestCase):
    def _skip_if_no_fp64(self, dtype):
        if dtype in FP64_DTYPES and not torch.xpu.get_device_properties(0).has_fp64:
            self.skipTest("device does not support fp64")

    @parametrize("dtype", DTYPES)
    def test_nonzero_passes(self, dtype):
        self._skip_if_no_fp64(dtype)
        torch._assert_async(torch.tensor(1, dtype=getattr(torch, dtype), device="xpu"))
        torch.xpu.synchronize()

    @parametrize("dtype", DTYPES)
    def test_zero_asserts(self, dtype):
        self._skip_if_no_fp64(dtype)
        p = subprocess.run(
            [sys.executable, "-c", SCRIPT.format(dtype=dtype)],
            capture_output=True,
            text=True,
        )
        output = p.stdout + "\n" + p.stderr
        self.assertNotEqual(p.returncode, 0, f"assert did not fire:\n{output}")
        self.assertIn("Assertion `input[0] != scalar_t(0)` failed", output)


instantiate_parametrized_tests(TestAssertAsync)


if __name__ == "__main__":
    run_tests()
