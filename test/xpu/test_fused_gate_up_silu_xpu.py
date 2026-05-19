# Owner(s): ["module: intel"]
import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_xpu import get_xpu_codename, XPUCodename


def check_if_pvc() -> bool:
    if not torch.xpu.is_available():
        return False
    xpu_codename = get_xpu_codename()
    if xpu_codename == XPUCodename.PVC:
        return True
    return False


class TestFusedGateUpSiLU(TestCase):
    """Correctness: compare fused kernel vs PyTorch reference."""

    def _reference(self, x, gate_w, up_w):
        return F.silu(x @ gate_w.T) * (x @ up_w.T)

    def _test_shape_dtype(self, device, M, K, N, dtype):
        x = torch.randn(M, K, device=device, dtype=dtype)
        gate_w = torch.randn(N, K, device=device, dtype=dtype)
        up_w = torch.randn(N, K, device=device, dtype=dtype)

        ref = self._reference(x, gate_w, up_w)
        out = torch.ops.xpu._fused_gate_up_silu(x, gate_w, up_w)

        rtol, atol = (2e-3, 0.5) if dtype == torch.float16 else (1.6e-2, 0.5)
        self.assertTrue(
            torch.allclose(ref, out, rtol=rtol, atol=atol),
            f"M={M} K={K} N={N} {dtype} max_diff={(ref - out).abs().max():.6e}",
        )

    @unittest.skipIf(check_if_pvc(), "PVC not available on sycl-tla fp16.")
    def test_fp16_shapes(self, device):
        for M in [1, 4, 32, 64, 128]:
            for K, N in [(512, 1384), (4096, 11008)]:
                self._test_shape_dtype(device, M, K, N, torch.float16)

    @unittest.skipIf(check_if_pvc(), "PVC not available on sycl-tla bf16.")
    def test_bf16_shapes(self, device):
        for M in [1, 4, 32, 64, 128]:
            self._test_shape_dtype(device, M, 512, 1384, torch.bfloat16)


instantiate_device_type_tests(
    TestFusedGateUpSiLU, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
