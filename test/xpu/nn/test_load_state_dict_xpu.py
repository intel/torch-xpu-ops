# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_load_state_dict import TestLoadStateDict, TestLoadStateDictSwap


instantiate_parametrized_tests(TestLoadStateDict)
instantiate_parametrized_tests(TestLoadStateDictSwap)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
