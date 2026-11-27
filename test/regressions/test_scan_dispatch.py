# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestScanDispatch(TestCase):
    @parametrize(
        "shape",
        [
            (1, 512),
            (1, 1100),
            (16, 32768),
            (64, 256),
            (64, 32768),
            (128, 512),
            (256, 32768),
        ],
    )
    @parametrize("dtype", [torch.uint8, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32])
    def test_cumsum_dispatch_boundaries(self, shape, dtype):
        if dtype.is_floating_point:
            x_cpu = (torch.rand(shape, dtype=torch.float32) * 0.1 - 0.05).to(dtype)
        else:
            x_cpu = torch.randint(0, 4, shape, dtype=dtype)
        x_xpu = x_cpu.to("xpu")

        expected = torch.cumsum(x_cpu, dim=1)
        actual = torch.cumsum(x_xpu, dim=1)
        if dtype == torch.bfloat16:
            ref = torch.cumsum(x_cpu.float(), dim=1).to(dtype)
            # 5-pass segment_scan on (16/64, 32768) rounds chunk outputs to bf16.
            atol = 0.5 if shape[1] >= 32768 else 2e-2
            self.assertEqual(actual.cpu(), ref, atol=atol, rtol=2e-2)
        elif dtype == torch.float16:
            ref = torch.cumsum(x_cpu.float(), dim=1).to(dtype)
            atol = 3e-2 if shape[1] >= 32768 else 1e-2
            self.assertEqual(actual.cpu(), ref, atol=atol, rtol=1e-2)
        else:
            self.assertEqual(actual.cpu(), expected)

    @parametrize("shape", [(1, 1100), (16, 4096), (64, 256), (256, 4096)])
    def test_cumprod_and_logcumsumexp_dispatch_boundaries(self, shape):
        # Keep magnitudes very close to 1 for cumprod stability on long rows.
        x_prod_cpu = 1.0 + (torch.rand(shape, dtype=torch.float32) - 0.5) * 1e-4
        self.assertEqual(
            torch.cumprod(x_prod_cpu.to("xpu"), dim=1).cpu(),
            torch.cumprod(x_prod_cpu, dim=1),
            atol=1e-4,
            rtol=1e-4,
        )

        x_lce_cpu = torch.randn(shape, dtype=torch.float32)
        self.assertEqual(
            torch.logcumsumexp(x_lce_cpu.to("xpu"), dim=1).cpu(),
            torch.logcumsumexp(x_lce_cpu, dim=1),
            atol=1e-4,
            rtol=1e-4,
        )


instantiate_parametrized_tests(TestScanDispatch)


if __name__ == "__main__":
    run_tests()
