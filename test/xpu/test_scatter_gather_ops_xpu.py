# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    import torch
    from test_scatter_gather_ops import TestScatterGather

    def _test_slice_scatter_compiled_backward_matches_cpu(self, device):
        def program(x, y):
            out = torch.slice_scatter(x, y, dim=1, start=0, end=6)
            return out.sum(dim=1)

        x = torch.randn(
            4, 13, 33, dtype=torch.float32, device=device, requires_grad=True
        )
        y = torch.randn(
            4, 6, 33, dtype=torch.float32, device=device, requires_grad=True
        )
        g = torch.randn(4, 33, dtype=torch.float32, device=device)

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

        self.assertEqual(out.cpu(), out_cpu, atol=1e-4, rtol=1e-4)
        self.assertEqual(x.grad.cpu(), x_cpu.grad, atol=1e-4, rtol=1e-4)
        self.assertEqual(y.grad.cpu(), y_cpu.grad, atol=1e-4, rtol=1e-4)

    TestScatterGather.test_slice_scatter_compiled_backward_matches_cpu = (
        _test_slice_scatter_compiled_backward_matches_cpu
    )


instantiate_device_type_tests(
    TestScatterGather, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
