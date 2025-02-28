# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


def rand_weight_uint8(n, k_div_2, device="xpu"):
    rand = torch.randint(0, 255, [n, k_div_2], device=device).to(torch.uint8)
    return rand


def weight_unpack(weight_packed, n, k_div_2):
    k_div_8 = k_div_2 // 4
    byte_d = (weight_packed & 0xFF000000) >> 24
    byte_c = (weight_packed & 0x00FF0000) >> 16
    byte_b = (weight_packed & 0x0000FF00) >> 8
    byte_a = weight_packed & 0x000000FF
    weight_unpacked = torch.empty([n, k_div_2], dtype=torch.uint8, device="xpu")
    weight_unpacked = weight_unpacked.view(n, k_div_8, 4)
    weight_unpacked[:, :, 0] = byte_a
    weight_unpacked[:, :, 1] = byte_b
    weight_unpacked[:, :, 2] = byte_c
    weight_unpacked[:, :, 3] = byte_d
    return weight_unpacked.view(n, -1)


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
