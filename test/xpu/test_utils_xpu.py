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
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_utils import (
        TestAssert,
        TestCheckpoint,
        TestCheckpointDeviceType,
        TestCollectEnv,
        TestCppExtensionUtils,
        TestDataLoaderUtils,
        TestDeprecate,
        TestDeviceLazyInit,
        TestDeviceUtils,
        TestHipify,
        TestHipifyTrie,
        TestRenderUtils,
        TestStandaloneCPPJIT,
        TestTraceback,
        TestTryImport,
    )


# --- TestTraceback overrides ---

# Fix regex in test_format_traceback_short to match test_utils_xpu.py filename.
# NOTE: Once we switch to the upstream file (test_utils.py), this override will
# no longer be needed — the upstream regex matches as-is.
from torch.utils._traceback import format_traceback_short


def _test_format_traceback_short(self):
    try:
        raise RuntimeError
    except RuntimeError as e:
        self.assertRegex(
            format_traceback_short(e.__traceback__),
            r".*test_utils(?:_xpu)?\.py:\d+ in _test_format_traceback_short",
        )


TestTraceback.test_format_traceback_short = _test_format_traceback_short


# --- TestCheckpointDeviceType overrides ---

# test_infer_device_state_recursive_multi_device: the upstream function
# _infer_device_type() uses sorted() on the device-type set, so the warning
# message lists types in alphabetical order.  Upstream asserts
#   f"Device types: ['{dev_type}', 'meta']"
# which works for CUDA ('cuda' < 'meta') but fails for XPU ('meta' < 'xpu').
# The old file-copy never hit this because TEST_MULTIGPU (CUDA-only) was False,
# so the test was always skipped.  Override to check both device types are
# present regardless of order.
import warnings

from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    onlyAccelerator,
)
from torch.utils.checkpoint import _infer_device_type


@onlyAccelerator
@deviceCountAtLeast(2)
def _test_infer_device_state_recursive_multi_device(self, devices):
    dev_type = torch.device(devices[0]).type
    inp = {
        "foo": torch.rand(10, device=f"{dev_type}:0"),
        "bar": [torch.rand(10, device=f"{dev_type}:1")],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _device_type = _infer_device_type(inp)
        self.assertEqual(dev_type, _device_type)
    inp = {
        "foo": torch.rand(10, device=f"{dev_type}:0"),
        "bar": [torch.rand(10, device=f"{dev_type}:0")],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _device_type = _infer_device_type(inp)
        self.assertEqual(dev_type, _device_type)
    inp = {
        "foo": torch.rand(10, device=f"{dev_type}:0"),
        "bar": [torch.rand(10, device="meta")],
    }
    with warnings.catch_warnings(record=True) as w:
        _device_type = _infer_device_type(inp)
        self.assertEqual(dev_type, _device_type)
    self.assertEqual(len(w), 1)
    warning_msg = str(w[-1].message)
    self.assertTrue(
        "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices"
        in warning_msg
    )
    # Check both device types present (order may vary by backend)
    self.assertIn(dev_type, warning_msg)
    self.assertIn("meta", warning_msg)
    self.assertTrue(f"first device type: {dev_type}" in warning_msg)


TestCheckpointDeviceType.test_infer_device_state_recursive_multi_device = (
    _test_infer_device_state_recursive_multi_device
)


instantiate_device_type_tests(
    TestCheckpointDeviceType, globals(), only_for="xpu", allow_xpu=True
)

instantiate_device_type_tests(
    TestDeviceUtils, globals(), only_for="xpu", allow_xpu=True
)

instantiate_device_type_tests(
    TestDeviceLazyInit, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
