# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


def rand_weight_uint8(n, k_div_2, device="xpu"):
    rand = torch.randint(-128, 128, [n, k_div_2], device=device).to(torch.uint8)
    return rand


def weight_unpack(weight_packed, n, k_div_2):
    k_div_8 = k_div_2 // 4
    weight_packed = weight_packed.view(n, k_div_8).contiguous()
    weight_packed = weight_packed.view(torch.uint8).view(n, k_div_8, 4)
    weight_unpacked = weight_packed.view(n, -1).contiguous()
    return weight_unpacked.view(torch.uint8).view(n, -1)


class TestNNMethod(TestCase):
    def _test_convert_weight_to_int4pack(self, n, k, device):
        k_div_2 = k // 2
        weight = rand_weight_uint8(n, k_div_2, device)
        weight_packed = torch.ops.aten._convert_weight_to_int4pack(weight, 8)
        weight_unpacked = weight_unpack(weight_packed, n, k_div_2)
        self.assertEqual(weight, weight_unpacked)

    def test_convert_weight_to_int4pack(self, device=torch.device("xpu")):
        for n in [512, 1024, 2048, 4096]:
            for k in [128, 4096]:
                self._test_convert_weight_to_int4pack(n, k, device)
