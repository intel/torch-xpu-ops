# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    serialTest,
    TestCase,
)


class TestForeachMetadata(TestCase):
    def _inputs(self, dtype, count, device):
        # Empty, single-element, and multi-chunk (131073 element) sizes
        # exercise zero, one, and many work-group table entries.
        sizes = (0, 1, 17, 4096, 131073)
        tensors = [
            torch.randn(sizes[i % len(sizes)], dtype=dtype, device=device)
            for i in range(count)
        ]
        return tensors, [torch.randn_like(t) for t in tensors]

    def _check(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_t, expected_t in zip(actual, expected):
            self.assertEqual(actual_t, expected_t)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("tensor_count", (3, 257))
    def test_foreach_add_tensor_list(self, device, dtype, tensor_count):
        a, b = self._inputs(dtype, tensor_count, device)
        expected = [x + y for x, y in zip(a, b)]

        out = torch._foreach_add(a, b)
        inplace = [x.clone() for x in a]
        torch._foreach_add_(inplace, b)
        # Queue consecutive calls before validating so the pinned host
        # metadata is reused while earlier copies may still be in flight.
        out_again = torch._foreach_add(a, b)
        torch.xpu.synchronize()

        self._check(out, expected)
        self._check(inplace, expected)
        self._check(out_again, expected)

    @dtypes(torch.float32, torch.float16, torch.bfloat16)
    @parametrize("tensor_count", (3, 257))
    def test_foreach_add_scalar_list(self, device, dtype, tensor_count):
        a, _ = self._inputs(dtype, tensor_count, device)
        scalars = [float(i % 7) / 2 for i in range(tensor_count)]
        expected = [x + scalar for x, scalar in zip(a, scalars)]

        out = torch._foreach_add(a, scalars)
        inplace = [x.clone() for x in a]
        torch._foreach_add_(inplace, scalars)
        out_again = torch._foreach_add(a, scalars)
        torch.xpu.synchronize()

        self._check(out, expected)
        self._check(inplace, expected)
        self._check(out_again, expected)

    @dtypes(torch.float32)
    def test_fused_adam_multi_chunk(self, device, dtype):
        # 131073 exceeds the fused chunk size (max work-group size * kILP), so
        # the fused optimizer path fills more than one work-group entry.
        params = [
            torch.randn(size, device=device, dtype=dtype, requires_grad=True)
            for size in (0, 1, 17, 131073)
        ]
        ref_params = [param.detach().clone().requires_grad_(True) for param in params]
        grads = [torch.randn_like(param) for param in params]
        for param, ref_param, grad in zip(params, ref_params, grads):
            param.grad = grad.clone()
            ref_param.grad = grad.clone()

        optimizer = torch.optim.Adam(params, lr=1e-3, fused=True)
        ref_optimizer = torch.optim.Adam(ref_params, lr=1e-3, fused=False)
        optimizer.step()
        ref_optimizer.step()

        for param, ref_param in zip(params, ref_params):
            self.assertEqual(param, ref_param)
            for state_key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(
                    optimizer.state[param][state_key],
                    ref_optimizer.state[ref_param][state_key],
                )

    @largeTensorTest("8GB")
    @serialTest()
    @parametrize("numel", (2**31 + 1024, 2**32))
    def test_foreach_large_numel(self, device, numel):
        x = torch.ones(numel, dtype=torch.int8, device=device)
        torch._foreach_add_([x], 1)
        self.assertEqual(x.min().item(), 2)
        self.assertEqual(x.max().item(), 2)

        torch._foreach_add_([x], [1])
        self.assertEqual(x.min().item(), 3)
        self.assertEqual(x.max().item(), 3)

        scalar = torch.tensor(1, dtype=torch.int8, device=device)
        torch._foreach_add_([x], scalar)
        self.assertEqual(x.min().item(), 4)
        self.assertEqual(x.max().item(), 4)

    @largeTensorTest("28GB")
    @serialTest()
    @parametrize("optim_cls", (torch.optim.SGD, torch.optim.Adagrad))
    def test_fused_optimizer_large_numel(self, device, optim_cls):
        p = torch.ones(2**32, dtype=torch.float16, device=device)
        p.grad = torch.ones_like(p)
        optim_cls([p], lr=0.01, fused=True).step()

        self.assertEqual(p.min(), p.max())
        ref = torch.ones(1, dtype=torch.float16, device=device)
        ref.grad = torch.ones_like(ref)
        optim_cls([ref], lr=0.01, fused=True).step()
        self.assertEqual(p[0], ref[0])


instantiate_device_type_tests(
    TestForeachMetadata, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
