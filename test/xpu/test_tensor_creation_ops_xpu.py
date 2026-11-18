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


import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfXPU,
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

# NOTE: `TestTensorCreationCudaOnly` is intentionally not imported: it is
# CUDA-specific and has no XPU counterpart.
with XPUImportCtx(False):
    # `TestTensorCreationGeneric` is built by an `@instantiate_parametrized_tests`
    # class decorator, which XPUImportCtx stubs out with a no-op that would
    # discard the class, so restore the real one for the duration of the import.
    common_utils.instantiate_parametrized_tests = instantiate_parametrized_tests
    from test_tensor_creation_ops import (
        TestAsArray,
        TestAsArrayCpuOnly,
        TestBufferProtocol,
        TestFromBlob,
        TestLikeTensorCreation,
        TestRandomTensorCreation,
        TestRandomTensorCreationCpuOnly,
        TestTensorCreation,
        TestTensorCreationGeneric,
    )


# ======================================================================
# Add dtypesIfXPU overrides for migrated tensor-creation tests
# ======================================================================

TestTensorCreation.test_signal_window_functions = dtypesIfXPU(
    torch.float, torch.double, torch.bfloat16, torch.half, torch.long
)(TestTensorCreation.test_signal_window_functions)

TestTensorCreation.test_logspace_device_vs_cpu = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestTensorCreation.test_logspace_device_vs_cpu)

TestTensorCreation.test_logspace_base2 = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestTensorCreation.test_logspace_base2)

TestTensorCreation.test_logspace_special_steps = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestTensorCreation.test_logspace_special_steps)

TestTensorCreation.test_logspace = dtypesIfXPU(
    *all_types_and(torch.half, torch.bfloat16)
)(TestTensorCreation.test_logspace)

TestRandomTensorCreation.test_uniform_from_to = dtypesIfXPU(
    torch.float, torch.double, torch.half, torch.bfloat16
)(TestRandomTensorCreation.test_uniform_from_to)


# ======================================================================
# Add XPU largeTensorTest override for migrated randperm coverage
# ======================================================================

TestRandomTensorCreation.test_randperm_large = largeTensorTest("40GB", "xpu")(
    TestRandomTensorCreation.test_randperm_large
)


# ======================================================================
# Add XPU-only large tensor creation tests
# ======================================================================


@dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
@largeTensorTest(
    lambda self, device, dtype: (2**31) * torch.tensor([], dtype=dtype).element_size()
)
def _test_zeros_large(self, device, dtype):
    _ = torch.zeros(2**31 - 1, device=device, dtype=dtype)


TestLikeTensorCreation.test_zeros_large = _test_zeros_large


@dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
@largeTensorTest(
    lambda self, device, dtype: (2**31) * torch.tensor([], dtype=dtype).element_size()
)
def _test_ones_large(self, device, dtype):
    _ = torch.ones(2**31 - 1, device=device, dtype=dtype)


TestLikeTensorCreation.test_ones_large = _test_ones_large


# ======================================================================
# Instantiate test classes for XPU execution
# ======================================================================

instantiate_device_type_tests(
    TestTensorCreation, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestRandomTensorCreation, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestLikeTensorCreation, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestRandomTensorCreationCpuOnly, globals(), only_for="cpu"
)
instantiate_device_type_tests(TestBufferProtocol, globals(), only_for="cpu")
instantiate_device_type_tests(TestFromBlob, globals(), only_for="cpu")
instantiate_device_type_tests(TestAsArray, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestAsArrayCpuOnly, globals(), only_for="cpu")


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
