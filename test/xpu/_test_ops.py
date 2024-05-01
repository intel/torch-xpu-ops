# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import run_tests

from xpu_test_utils import XPUTestPatch, instantiate_device_type_tests

with XPUTestPatch():
    from test_ops import TestCommon, TestMathBits, TestRefsOpsInfo, TestFakeTensor, TestTags


instantiate_device_type_tests(TestCommon, globals(), only_for="xpu")
instantiate_device_type_tests(TestMathBits, globals(), only_for="xpu")
instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for="xpu")
instantiate_device_type_tests(TestFakeTensor, globals(), only_for="xpu")
instantiate_device_type_tests(TestTags, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
