# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


class TestOperationOnDevice1(TestCase):
    def test_sum_on_device1(self, dtype=torch.float):
        if torch.xpu.device_count() >= 2:
            a = torch.randn(2, 3, device=torch.device("xpu:1"))
            torch.xpu.set_device(1)
            res = a.sum()
            ref = a.cpu().sum()
            self.assertEqual(ref, res)
