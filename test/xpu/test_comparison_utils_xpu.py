# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_comparison_utils import TestComparisonUtils


def _test_assert_device(self):
    t = torch.tensor([0.5], device="cpu")

    with self.assertRaises(RuntimeError):
        torch._assert_tensor_metadata(t, device="xpu")


TestComparisonUtils.test_assert_device = _test_assert_device


if __name__ == "__main__":
    run_tests()
