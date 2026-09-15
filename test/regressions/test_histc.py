# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestHistc(TestCase):
    def test_histc_out_rejects_mismatched_dtype(self):
        # Regression for intel/torch-xpu-ops#5237: _histc_out_xpu finished
        # with result.copy_(ret), so a float histogram written into an
        # integral out= was silently truncated. CPU rejects this call.
        x = torch.linspace(1, 8, 8, device="xpu", dtype=torch.float32)
        out = torch.empty(4, device="xpu", dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "should have the same dtype"):
            torch.histc(x, bins=4, min=0, max=8, out=out)

    def test_histc_out_matching_dtype_still_works(self):
        x = torch.linspace(1, 8, 8, device="xpu", dtype=torch.float32)
        out = torch.empty(0, device="xpu", dtype=torch.float32)

        torch.histc(x, bins=4, min=0, max=8, out=out)
        self.assertEqual(out, torch.histc(x, bins=4, min=0, max=8))


if __name__ == "__main__":
    run_tests()
