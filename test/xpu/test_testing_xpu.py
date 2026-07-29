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

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    import test_testing as _test_testing_mod
    from test_testing import (
        expectedFailureMeta,
        onlyNativeDeviceTypes,
        op_db,
        opinfo,
        TestAssertClose,
        TestAssertCloseContainer,
        TestAssertCloseErrorMessage,
        TestAssertCloseMultiDevice,
        TestAssertCloseQuantized,
        TestAssertCloseSparseBSC,
        TestAssertCloseSparseBSR,
        TestAssertCloseSparseCOO,
        TestAssertCloseSparseCSC,
        TestAssertCloseSparseCSR,
        TestEnvironmentDefFlag,
        TestFrameworkUtils,
        TestImports,
        TestMakeTensor,
        TestOpInfos,
        TestOpInfoSampleFunctions,
        TestTesting,
        TestTestParametrization,
        TestTestParametrizationDeviceType,
    )

# ======================================================================
# Restore instantiate_* bindings in the imported module
# test_testing.py imports instantiate_parametrized_tests and
# instantiate_device_type_tests at module level.  During import under
# XPUImportCtx these were patched to DO_NOTHING, so the module-local
# names still reference the no-op.  Several test methods call these
# functions on locally-defined classes and assert the results — they
# would silently fail without this fixup.
# ======================================================================

_test_testing_mod.instantiate_parametrized_tests = instantiate_parametrized_tests
_test_testing_mod.instantiate_device_type_tests = instantiate_device_type_tests


# ======================================================================
# XPU-specific method override: test_get_supported_dtypes
# Upstream only handles cpu/cuda.  Add the xpu branch so
# dtypesIfXPU is checked when running on XPU device.
# ======================================================================


@expectedFailureMeta
@onlyNativeDeviceTypes
def _test_get_supported_dtypes(self, device):
    ops_to_test = list(
        filter(lambda op: op.formatted_name in ["atan2", "topk", "xlogy"], op_db)
    )

    for op in ops_to_test:
        dynamic_dtypes = opinfo.utils.get_supported_dtypes(
            op, op.sample_inputs_func, self.device_type
        )
        dynamic_dispatch = opinfo.utils.dtypes_dispatch_hint(dynamic_dtypes)
        if self.device_type == "cpu":
            dtypes = op.dtypes
        elif self.device_type == "xpu":
            dtypes = op.dtypesIfXPU
        else:
            dtypes = op.dtypesIfCUDA

        self.assertTrue(set(dtypes) == set(dynamic_dtypes))
        self.assertTrue(set(dtypes) == set(dynamic_dispatch.dispatch_fn()))


TestTesting.test_get_supported_dtypes = _test_get_supported_dtypes


# ======================================================================
# Instantiate test classes
# ======================================================================
# NOTE: The `TestTesting` and `TestTestParametrizationDeviceType` classes are
# not parametrized for XPU.  When adding `allow_xpu=True` several test cases fail.
# When upstreaming this needs to be investigated further.

instantiate_device_type_tests(TestTesting, globals())
instantiate_device_type_tests(
    TestAssertCloseMultiDevice, globals(), only_for=("cuda", "xpu"), allow_xpu=True
)
instantiate_device_type_tests(TestMakeTensor, globals(), allow_xpu=True)
instantiate_parametrized_tests(TestTestParametrization)
instantiate_device_type_tests(TestTestParametrizationDeviceType, globals())
instantiate_device_type_tests(TestOpInfoSampleFunctions, globals(), allow_xpu=True)
instantiate_parametrized_tests(TestImports)


if __name__ == "__main__":
    run_tests()
