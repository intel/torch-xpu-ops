# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import TestCase


class TestDivMode(TestCase):
    def test_div_true_dtype(self):
        claimed_dtypes = get_all_dtypes()
        for dtype in claimed_dtypes:
            a_cpu = torch.randint(1, 100, [8, 8]).to(dtype)
            a_xpu = a_cpu.to("xpu")
            ref = torch.ops.aten.div(a_cpu * 2, a_cpu, rounding_mode=None)
            res = torch.ops.aten.div(a_xpu * 2, a_xpu, rounding_mode=None)
            self.assertEqual(ref, res.to("cpu"))
