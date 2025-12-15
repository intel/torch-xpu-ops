# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

from torch.nn.parameter import UninitializedParameter
from torch.testing._internal.common_utils import run_tests, suppress_warnings

try:
    from .xpu_test_utils import XPUPatchForImport
except Exception as e:
    from ..xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_lazy_modules import LazyModule, TestLazyModules


@suppress_warnings
def materialize_device(self):
    module = LazyModule()
    module.register_parameter("test_param", UninitializedParameter())
    module.test_param.materialize(10)
    self.assertTrue(module.test_param.device.type == "cpu")
    device = "xpu"
    module = LazyModule()
    module.register_parameter("test_param", UninitializedParameter())
    module.to(device)
    module.test_param.materialize(10)
    self.assertTrue(module.test_param.device.type == device)


TestLazyModules.test_materialize_device = materialize_device

if __name__ == "__main__":
    run_tests()
