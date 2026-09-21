# Owner(s): ["module: intel"]

import math

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestXpuTransposeAlignment(TestCase):
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_transpose_copy_unaligned(self, device, dtype):
        shape = (16, 64, 32, 32)

        def offset_view(offset, view_shape):
            base = torch.empty(
                offset + math.prod(shape), device=device, dtype=dtype
            )
            return base[offset:].view(view_shape)

        src = offset_view(1, (16, 32, 32, 64)).permute(0, 3, 1, 2)
        dst = offset_view(1, shape)
        src.copy_(torch.randn(shape, device=device, dtype=dtype))

        self.assertNotEqual(src.data_ptr() % (4 * src.element_size()), 0)
        self.assertNotEqual(dst.data_ptr() % (4 * dst.element_size()), 0)
        dst.copy_(src)

        self.assertEqual(dst, src)


instantiate_device_type_tests(
    TestXpuTransposeAlignment, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
