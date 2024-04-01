# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import run_tests

import test_xpu
from test_xpu import instantiate_device_type_tests
from test_ops import TestCommon

instantiate_device_type_tests(TestCommon, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
