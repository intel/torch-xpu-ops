# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: complex"]
# ruff: noqa: F841

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from ..xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_complex_tensor import TestComplexBwdGradients, TestComplexTensor

    if dist.is_available():
        from test_complex_tensor import TestComplexDistributed


instantiate_device_type_tests(
    TestComplexTensor, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestComplexBwdGradients, globals(), only_for="xpu", allow_xpu=True
)


if dist.is_available():
    instantiate_device_type_tests(
        TestComplexDistributed, globals(), only_for="xpu", allow_xpu=True
    )

if __name__ == "__main__":
    run_tests()
