import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="requires XPU",
)
def test_adaptive_avg_pool2d_flatten_sum_matches_eager():
    def fn(x):
        y = F.adaptive_avg_pool2d(x, 7)
        return y.flatten(1).sum(dim=-1)

    torch.manual_seed(42)
    x_cpu = torch.randn(2, 33, 8, 8, dtype=torch.float64)
    x_xpu = x_cpu.to("xpu")

    cpu_out = fn(x_cpu)
    xpu_eager = fn(x_xpu).cpu()

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor")
    xpu_compiled = compiled(x_xpu).cpu()

    torch.testing.assert_close(xpu_eager, cpu_out, rtol=0, atol=1e-12)
    torch.testing.assert_close(xpu_compiled, cpu_out, rtol=0, atol=1e-12)
