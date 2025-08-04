# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


class TestSimpleBinary(TestCase):
    def test_tril(self, dtype=torch.bool):
        max_seq_length = 131072
        with torch.device("xpu"):
            torch.xpu.empty_cache()
            a = torch.ones(max_seq_length, max_seq_length, dtype=torch.bool)
            causal_mask = torch.tril(a)
        torch.xpu.synchronize()
        print(torch.xpu.get_device_properties())
