# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(not hasattr(torch, "xpu"), reason="requires torch.xpu")
@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
def test_slice_scatter_xpu_compile_backward_matches_cpu():
    def program(x, y):
        out = torch.slice_scatter(x, y, dim=1, start=0, end=6)
        return out.sum(dim=1)

    torch.manual_seed(0)
    x = torch.randn(4, 13, 33, dtype=torch.float32, device="xpu", requires_grad=True)
    y = torch.randn(4, 6, 33, dtype=torch.float32, device="xpu", requires_grad=True)
    g = torch.randn(4, 33, dtype=torch.float32, device="xpu")

    torch._dynamo.reset()
    try:
        compiled = torch.compile(program, backend="inductor")
        out = compiled(x, y)
        out.backward(g)
    finally:
        torch._dynamo.reset()

    x_cpu = x.detach().cpu().requires_grad_(True)
    y_cpu = y.detach().cpu().requires_grad_(True)
    g_cpu = g.detach().cpu()
    out_cpu = program(x_cpu, y_cpu)
    out_cpu.backward(g_cpu)

    assert torch.allclose(out.cpu(), out_cpu, atol=1e-4, rtol=1e-4)
    assert torch.allclose(x.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(y.grad.cpu(), y_cpu.grad, atol=1e-4, rtol=1e-4)
