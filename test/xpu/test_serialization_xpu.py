# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    run_tests,
    TemporaryFileName,
)
from torch.serialization import safe_globals

try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_serialization import (
        TestBothSerialization,
        TestSerialization,
        TestSubclassSerialization,
    )
    from torch.testing._internal.two_tensor import TwoTensor


# Override TestCase-level (non-device) tests that hardcode cuda; mirror upstream
# bodies on the xpu device. Required because XPUPatchForImport leaves @unittest.skipIf(
# not torch.cuda.is_available()) intact, which auto-skips on XPU machines.
def _xpu_test_serialization_mmap_loading_with_map_location(self):
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = torch.nn.Linear(3, 1024)
            self.fc2 = torch.nn.Linear(1024, 5)

        def forward(self, input):
            return self.fc2(self.fc1(input))

    with TemporaryFileName() as f:
        with torch.device("xpu"):
            m = DummyModel()
        state_dict = m.state_dict()
        torch.save(state_dict, f)
        result = torch.load(f, mmap=True)
        for v in result.values():
            self.assertTrue(v.is_xpu)


TestSerialization.test_serialization_mmap_loading_with_map_location = (
    _xpu_test_serialization_mmap_loading_with_map_location
)


def _xpu_test_tensor_subclass_map_location(self):
    t = TwoTensor(torch.randn(2, 3), torch.randn(2, 3))
    sd = {"t": t}

    with TemporaryFileName() as f:
        torch.save(sd, f)
        with safe_globals([TwoTensor]):
            sd_loaded = torch.load(f, map_location=torch.device("xpu:0"))
            self.assertTrue(sd_loaded["t"].device == torch.device("xpu:0"))


TestSubclassSerialization.test_tensor_subclass_map_location = (
    _xpu_test_tensor_subclass_map_location
)


instantiate_device_type_tests(
    TestBothSerialization, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
