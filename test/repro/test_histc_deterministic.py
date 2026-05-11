# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
import pytest
from torch.testing._internal.common_utils import TestCase


class TestHistcDeterministic(TestCase):
    def test_histc_int_deterministic_mode(self):
        """histc with integer input must not raise an error in deterministic mode."""
        nbins = 8
        int_tensor = torch.randint(0, nbins, (100,), device="xpu")

        torch.use_deterministic_algorithms(True)
        try:
            result = torch.histc(int_tensor, bins=nbins, min=0, max=nbins - 1)
            # result should have the correct number of bins
            self.assertEqual(result.numel(), nbins)
        finally:
            torch.use_deterministic_algorithms(False)

    def test_histc_float_deterministic_mode_raises(self):
        """histc with float input must raise an error in deterministic mode."""
        nbins = 8
        float_tensor = torch.randint(0, nbins, (100,), device="xpu").float()

        torch.use_deterministic_algorithms(True)
        try:
            with self.assertRaisesRegex(RuntimeError, "deterministic"):
                torch.histc(float_tensor, bins=nbins, min=0, max=nbins - 1)
        finally:
            torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
