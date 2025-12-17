# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import sys

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import NoTest, run_tests, TEST_XPU, TestCase

try:
    from xpu_test_utils import copy_tests, XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import copy_tests, XPUPatchForImport

with XPUPatchForImport():
    from test_tensor_creation_ops import TestTensorCreation as TestTensorCreationBase

if not TEST_XPU:
    print("XPU not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811

TEST_MULTIXPU = torch.xpu.device_count() > 1

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestTensorCreationXPU(TestCase):
    pass


copy_tests(
    TestTensorCreationXPU,
    TestTensorCreationBase,
    applicable_list=["test_empty_strided"],
)
instantiate_device_type_tests(
    TestTensorCreationXPU, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
