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
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPinMemory(TestCase):
    def test_pin_memory(self, device):
        a = torch.randn(100)
        self.assertFalse(a.is_pinned())
        b = a.pin_memory()
        self.assertTrue(b.is_pinned())
        x = a.to(device)
        self.assertFalse(x.is_pinned())
        with self.assertRaisesRegex(
            RuntimeError,
            "only dense CPU tensors can be pinned",
        ):
            y = x.pin_memory()
            del y


instantiate_device_type_tests(TestPinMemory, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
