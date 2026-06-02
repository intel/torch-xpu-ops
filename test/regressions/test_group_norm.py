# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import math
from functools import partial

import torch
from torch.testing._internal.common_utils import TestCase


class TestGroupNorm(TestCase):
    """Regression test for native_group_norm fp16 accuracy on XPU.

    Verifies eager and torch.compile produce consistent results.
    See https://github.com/pytorch/pytorch/issues/185367.
    """

    def _make_sample_inputs(self):
        """Generate the same sample inputs as sample_inputs_native_group_norm."""
        device = "xpu"
        dtype = torch.float16
        make_arg = partial(
            torch.testing.make_tensor, device=device, dtype=dtype, requires_grad=False
        )

        # Same cases as sample_inputs_group_norm
        cases = (
            ((1, 6, 3), 2, {"eps": 0.5}),
            ((2, 6, 3), 2, {"eps": -0.5}),
            ((1, 3), 1, {"eps": 1e-5}),
            ((0, 2), 1, {"eps": 1e-5}),
            ((5, 5, 5), 1, {"eps": 0.5}),
        )

        samples = []
        for input_shape, num_groups, kw in cases:
            channels = input_shape[1] if len(input_shape) > 1 else 0
            weight_tensor = make_arg(channels)
            bias_tensor = make_arg(channels)

            for weight in [weight_tensor, None]:
                for bias in [bias_tensor, None]:
                    inp = make_arg(input_shape)
                    N = inp.shape[0]
                    C = inp.shape[1]
                    HxW = math.prod(inp.shape[2:])
                    eps = kw["eps"]
                    samples.append((inp, weight, bias, N, C, HxW, num_groups, eps))

        # Without any optional args
        inp = make_arg((1, 2))
        samples.append((inp, None, None, 1, 2, 1, 1, 1e-5))
        return samples

    def test_group_norm_eager_vs_compile_float16(self):
        @torch.compile
        def compiled_group_norm(x, weight, bias, N, C, HxW, group, eps):
            return torch.native_group_norm(x, weight, bias, N, C, HxW, group, eps)

        for inp, weight, bias, N, C, HxW, group, eps in self._make_sample_inputs():
            eager_out = torch.native_group_norm(
                inp, weight, bias, N, C, HxW, group, eps
            )
            compiled_out = compiled_group_norm(inp, weight, bias, N, C, HxW, group, eps)
            # Same tolerance as TestInductorOpInfoXPU: float16 default
            for e, c in zip(eager_out, compiled_out):
                self.assertEqual(e, c)
