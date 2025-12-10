# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_maskedtensor import (
        TestBasics,
        TestBinary,
        TestOperators,
        TestReductions,
        TestUnary,
    )

instantiate_device_type_tests(TestBasics, globals(), only_for=("xpu"), allow_xpu=True)

instantiate_device_type_tests(
    TestOperators, globals(), only_for=("xpu"), allow_xpu=True
)
instantiate_parametrized_tests(TestUnary)
instantiate_parametrized_tests(TestBinary)
instantiate_parametrized_tests(TestReductions)

if __name__ == "__main__":
    run_tests()
