# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase

device = torch.device("xpu")
cpu_device = torch.device("cpu")


class TestTorchMethod(TestCase):
    def test_upsample_bilinear_bwd(self):
        test_dtypes = [torch.float]

        def _test_upsample_bilinear_bwd(dtype):
            grad_output_cpu = torch.randn(
                (4, 4, 512, 512), dtype=dtype, device=cpu_device
            )
            for memory_format in [torch.channels_last, torch.contiguous_format]:
                for align_corners in [True, False]:
                    grad_output_cpu = grad_output_cpu.contiguous(
                        memory_format=memory_format
                    )
                    r_cpu = torch._ops.ops.aten.upsample_bilinear2d_backward(
                        grad_output_cpu.to(torch.float64),
                        output_size=(512, 512),
                        input_size=(4, 4, 256, 256),
                        align_corners=align_corners,
                    )
                    r_xpu = torch._ops.ops.aten.upsample_bilinear2d_backward(
                        grad_output_cpu.to("xpu"),
                        output_size=(512, 512),
                        input_size=(4, 4, 256, 256),
                        align_corners=align_corners,
                    )

                    self.assertEqual(r_cpu.to(dtype), r_xpu.cpu())

        for dtype in test_dtypes:
            _test_upsample_bilinear_bwd(dtype)
