# Owner(s): ["module: intel"]

import test_xpu

test_xpu.create_template("test_ops", "test_ops_template")
from test_ops_template import TestCommon

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests

instantiate_device_type_tests(TestCommon, globals(), only_for="xpu")


if __name__ == "__main__":
    run_tests()
