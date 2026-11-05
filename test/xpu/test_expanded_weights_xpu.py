# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import copy

import torch.nn as nn
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import get_new_module_tests, module_tests
from torch.testing._internal.common_utils import run_tests, TEST_XPU

# Save a copy before the upstream import mutates the dicts
# (it pops "module_name" and "decorator", and sets "constructor").
_module_tests_copy = copy.deepcopy(module_tests)

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    from test_expanded_weights import (
        ContextManagerTests,
        filter_supported_tests,
        TestExpandedWeightFunctional,
        TestExpandedWeightHelperFunction,
        TestExpandedWeightModule,
    )


# Add test_xpu support (upstream only defines test_cpu and test_cuda).
def _cm_init(self, *args, **kwargs):
    self.test_cpu = kwargs.get("test_cpu", True)
    self.test_cuda = kwargs.get("test_cuda", True)
    self.test_xpu = kwargs.get("test_xpu", True)
    super(ContextManagerTests, self).__init__(*args, **kwargs)


ContextManagerTests.__init__ = _cm_init

# Upstream only registers _cuda_double methods on TestExpandedWeightModule.
# Re-run the same loop for XPU (mirroring test/test_expanded_weights.py)
# and remove the _cuda_double originals.
_supported_tests = [
    t for t in _module_tests_copy + get_new_module_tests() if filter_supported_tests(t)
]
for test_param in _supported_tests:
    if "constructor" not in test_param:
        name = test_param.pop("module_name")
        test_param["constructor"] = getattr(nn, name)
    decorator = test_param.pop("decorator", lambda test: test)
    test = ContextManagerTests(**test_param)
    test_name = test.get_name()
    # Remove the _cuda_double original (hardcodes device="cuda").
    if hasattr(TestExpandedWeightModule, test_name + "_cuda_double"):
        delattr(TestExpandedWeightModule, test_name + "_cuda_double")
    if TEST_XPU and test.test_xpu:
        # since this checks derivatives, only use double for precision
        setattr(
            TestExpandedWeightModule,
            test_name + "_xpu_double",
            decorator(lambda self, test=test: test.test_context_manager(self, "xpu")),
        )

instantiate_device_type_tests(
    TestExpandedWeightHelperFunction, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestExpandedWeightFunctional, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestExpandedWeightModule, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
