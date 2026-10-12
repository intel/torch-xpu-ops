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
# ruff: noqa: F401

import inspect
import unittest

import numpy as np
import torch
from torch import Tensor
from torch.testing._internal import custom_op_db
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    skipIfTorchDynamo,
    TEST_ACCELERATOR,
    TEST_XPU,
    TestCase,
)

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_custom_ops import (
        MiniOpTest,
        MiniOpTestOther,
        TestCustomOp,
        TestCustomOpAPI,
        TestCustomOpTesting,
        TestGenerateOpcheckTests,
        TestLibrarySourceLocation,
        TestOpProfiles,
        TestTypeConversion,
    )


# ======================================================================
# Method overrides (body changes for device generalization)
# ======================================================================


@skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
@unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator")
def _test_split_device(self):
    cpu_call_count = 0
    accelerator_call_count = 0
    device_type = torch.accelerator.current_accelerator().type

    @torch.library.custom_op("_torch_testing::f", mutates_args=(), device_types="cpu")
    def f(x: Tensor) -> Tensor:
        nonlocal cpu_call_count
        cpu_call_count += 1
        x_np = x.numpy()
        out_np = np.sin(x_np)
        return torch.from_numpy(out_np)

    @f.register_kernel(device_type)
    def _(x: Tensor) -> Tensor:
        nonlocal accelerator_call_count
        accelerator_call_count += 1
        x_np = x.cpu().numpy()
        out_np = np.sin(x_np)
        return torch.from_numpy(out_np).to(x.device)

    x = torch.randn(3)
    y = f(x)
    self.assertEqual(y, x.sin())
    self.assertEqual(cpu_call_count, 1)
    self.assertEqual(accelerator_call_count, 0)

    x = x.to(device_type)
    y = f(x)
    self.assertEqual(y, x.sin())
    self.assertEqual(cpu_call_count, 1)
    self.assertEqual(accelerator_call_count, 1)


TestCustomOpAPI.test_split_device = _test_split_device


@skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
@unittest.skipIf(not TEST_ACCELERATOR, "requires accelerator")
def _test_multi_types(self):
    device_type = torch.accelerator.current_accelerator().type

    @torch.library.custom_op(
        "_torch_testing::f",
        mutates_args=(),
        device_types=("cpu", device_type),
    )
    def f(x: Tensor) -> Tensor:
        x_np = x.cpu().numpy()
        out_np = np.sin(x_np)
        return torch.from_numpy(out_np).to(x.device)

    x = torch.randn(3)
    y = f(x)
    self.assertEqual(y, x.sin())
    x = x.to(device_type)
    y = f(x)
    self.assertEqual(y, x.sin())


TestCustomOpAPI.test_multi_types = _test_multi_types

# ======================================================================
# Test opcheck method override for TestCustomOpTesting
# ======================================================================


@ops(custom_op_db.custom_op_db, dtypes=OpDTypes.any_one)
def _test_opcheck_opinfo(self, device, dtype, op):
    for sample_input in op.sample_inputs(
        device, dtype, requires_grad=op.supports_autograd
    ):
        args = [sample_input.input] + list(sample_input.args)
        kwargs = sample_input.kwargs
        torch.library.opcheck(op.op, args, kwargs)


TestCustomOpTesting.test_opcheck_opinfo = _test_opcheck_opinfo


# ======================================================================
# Decorator additions (no body change)
# ======================================================================


run_on_accelerator = unittest.skipUnless(
    TEST_ACCELERATOR, "pinned CPU memory requires CUDA or XPU"
)

TestGenerateOpcheckTests.test_opcheck_preserves_pinned_memory_by_default = (
    run_on_accelerator(
        inspect.unwrap(
            TestGenerateOpcheckTests.test_opcheck_preserves_pinned_memory_by_default
        )
    )
)

TestGenerateOpcheckTests.test_opcheck_preserves_pinned_memory_for_schema_check = run_on_accelerator(
    inspect.unwrap(
        TestGenerateOpcheckTests.test_opcheck_preserves_pinned_memory_for_schema_check
    )
)

TestGenerateOpcheckTests.test_safe_schema_check_copy_inputs_preserves_pinned_memory_and_copies = run_on_accelerator(
    inspect.unwrap(
        TestGenerateOpcheckTests.test_safe_schema_check_copy_inputs_preserves_pinned_memory_and_copies
    )
)


# ======================================================================
# Instantiate tests for XPU
# ======================================================================

instantiate_device_type_tests(
    TestCustomOpTesting, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestCustomOp)
instantiate_parametrized_tests(TestCustomOpAPI)

if __name__ == "__main__":
    run_tests()
