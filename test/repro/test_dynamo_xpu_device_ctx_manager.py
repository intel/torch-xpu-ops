# Owner(s): ["module: dynamo"]
import pytest
import torch
from torch.testing._internal.common_utils import run_tests


@pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(), reason="requires dynamo")
@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires xpu")
def test_dynamo_xpu_device_ctx_manager():
    def fn(x):
        safe_device_index = max(x.device.index - 1, 0)
        with torch.xpu.device(safe_device_index):
            return torch.sin(x + 1)

    x = torch.randn((2, 2), device="xpu")
    ref = fn(x)
    with torch._dynamo.dont_skip_tracing():
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x)
    torch.testing.assert_close(ref, res)


if __name__ == "__main__":
    run_tests()
