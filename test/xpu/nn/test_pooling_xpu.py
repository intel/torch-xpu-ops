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
    from .xpu_test_utils import XPUImportCtx
except Exception as e:
    from ..xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_pooling import (
        TestAvgPool,
        TestAvgPoolDevice,
        TestPoolingNN,
        TestPoolingNNDevice,
    )


instantiate_device_type_tests(
    TestAvgPoolDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestPoolingNNDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestPoolingNN)


if __name__ == "__main__":
    run_tests()
