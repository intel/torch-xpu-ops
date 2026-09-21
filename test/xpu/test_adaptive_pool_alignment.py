# Owner(s): ["module: intel"]

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


def offset_channels_last(shape, dtype, offset):
    n, c, h, w = shape
    base = torch.empty(offset + n * c * h * w, device="xpu", dtype=dtype)
    return base[offset:].view(n, h, w, c).permute(0, 3, 1, 2)


class TestXpuAdaptivePoolAlignment(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_adaptive_avg_pool2d_unaligned_channels_last(self, device, dtype):
        shape = (8, 64, 32, 32)
        input = offset_channels_last(shape, dtype, 1)
        input.copy_(torch.randn(shape, device=device, dtype=dtype))

        self.assertNotEqual(input.data_ptr() % (4 * input.element_size()), 0)
        actual = F.adaptive_avg_pool2d(input, (7, 9))
        expected = F.adaptive_avg_pool2d(input.cpu(), (7, 9))

        self.assertEqual(actual.cpu(), expected)


instantiate_device_type_tests(
    TestXpuAdaptivePoolAlignment, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
