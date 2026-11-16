# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Regression tests for index bounds checks in XPU indexing ops.

- Deterministic index_add_ goes through index_put_, which accepts negative
  indices, so -1 silently added to the last row.

A failing bounds check aborts the process through SYCL_KERNEL_ASSERT, so every
case runs in its own subprocess. Each case first proves the device is healthy,
and a separate process checks it again afterwards. Inputs are 1-D so a single
work-item fails the assert.
"""

import subprocess
import sys

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)

TIMEOUT_S = 120

SCRIPT = """
import torch
torch.ones(1, device="xpu").add_(1)
torch.xpu.synchronize()
print("device ok", flush=True)
torch.use_deterministic_algorithms({det})
x = torch.zeros(4, device="xpu")
src = torch.ones(1, device="xpu")
idx = torch.tensor([{index}], device="xpu")
{op}
torch.xpu.synchronize()
"""

HEALTH = """
import torch
x = torch.ones(4, device="xpu")
torch.xpu.synchronize()
assert x.sum().item() == 4
"""

VALID = """
import torch

def run(device, det):
    torch.use_deterministic_algorithms(det)
    x = torch.arange(4.0, device=device)
    src = torch.ones(2, device=device)
    idx = torch.tensor([0, 3], device=device)
    out = [x.clone().index_add_(0, idx, src), x.clone().index_add_(0, idx.int(), src)]
    if det:
        b = torch.zeros(4, dtype=torch.bool, device=device)
        out.append(b.index_add_(0, idx, torch.ones(2, dtype=torch.bool, device=device), alpha=True))
    return out

for det in (False, True):
    for expected, actual in zip(run("cpu", det), run("xpu", det)):
        torch.testing.assert_close(actual.cpu(), expected)
torch.xpu.synchronize()
"""

DETERMINISTIC_VALIDATION = """
import torch
torch.use_deterministic_algorithms(True)
x = torch.tensor({self_value}, dtype=torch.{dtype}, device="xpu")
src = torch.tensor({source_value}, dtype=torch.{dtype}, device="xpu")
idx = torch.tensor([-1], device="xpu")
try:
    x.index_add_(0, idx, src, alpha={alpha})
except Exception as e:
    print(type(e).__name__, str(e), flush=True)
else:
    raise AssertionError("expected a validation error")
torch.xpu.synchronize()
print("synchronized", flush=True)
"""

INDEX_ADD = "x.index_add_(0, idx, src)"

# Expected SYCL_KERNEL_ASSERT conditions.
RANGE_CHECK = "idx >= 0 && idx < self_dim_size_"

BOOL_MISMATCH = "RuntimeError expected scalar type Long but found Bool"
INT_MISMATCH = "RuntimeError expected scalar type Float but found Int"
SCALAR_INDEX_ERROR = "IndexError too many indices for tensor of dimension 0"

# op, deterministic, index, expected assert condition.
CASES = [
    subtest((INDEX_ADD, True, -1, RANGE_CHECK), name="index_add_det_neg1"),
    subtest((INDEX_ADD, True, 4, RANGE_CHECK), name="index_add_det_too_large"),
]


class TestIndexOpsIndexBounds(TestCase):
    def _run(self, script):
        cmd = [sys.executable, "-c", script]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            self.fail(f"timed out after {TIMEOUT_S}s:\n{e.stdout}\n{e.stderr}")
        return p.returncode, p.stdout + "\n" + p.stderr

    @parametrize("op, det, index, condition", CASES)
    def test_invalid_index_is_rejected(self, op, det, index, condition):
        rc, output = self._run(SCRIPT.format(op=op, det=det, index=index))
        self.assertIn("device ok", output, f"unhealthy before the case:\n{output}")
        self.assertNotEqual(rc, 0, f"{op} accepted index {index}:\n{output}")
        self.assertIn(f"Assertion `{condition}` failed", output)
        rc, output = self._run(HEALTH)
        self.assertEqual(rc, 0, f"device did not recover:\n{output}")

    def test_valid_indices_match_cpu(self):
        rc, output = self._run(VALID)
        self.assertEqual(rc, 0, output)

    @parametrize(
        "dtype, self_value, source_value, alpha, error",
        [
            subtest(("bool", [0] * 4, [1], 1, BOOL_MISMATCH), name="bool"),
            subtest(("int32", [0] * 4, [1], 0.5, INT_MISMATCH), name="promoted_alpha"),
            subtest(
                ("uint16", [0] * 4, [1], 1, 'NotImplementedError "mul_xpu"'),
                name="unsupported_mul",
            ),
            subtest(("float32", 0, 1, 1, SCALAR_INDEX_ERROR), name="scalar"),
        ],
    )
    def test_deterministic_validation_before_bounds_check(
        self, dtype, self_value, source_value, alpha, error
    ):
        script = DETERMINISTIC_VALIDATION.format(
            dtype=dtype,
            self_value=self_value,
            source_value=source_value,
            alpha=alpha,
        )
        rc, output = self._run(script)
        self.assertEqual(rc, 0, output)
        self.assertIn(error, output)
        self.assertIn("synchronized", output)


instantiate_parametrized_tests(TestIndexOpsIndexBounds)


if __name__ == "__main__":
    run_tests()
