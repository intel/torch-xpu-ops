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
import os
import sys

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import instantiate_parametrized_tests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx


with XPUImportCtx(False):
    from dynamo.test_misc import (
        DynamoOpPromotionTests,
        MiscTests,
        MiscTestsDevice,
        MiscTestsPyTree,
        TestCustomFunction,
        TestTracer,
    )


instantiate_parametrized_tests(MiscTests)
instantiate_parametrized_tests(MiscTestsPyTree)

instantiate_device_type_tests(
    MiscTestsDevice, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
