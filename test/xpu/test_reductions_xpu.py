# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import numpy as np
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_dtype import all_types_and
from torch.testing._internal.common_utils import make_tensor, run_tests, TestCase

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_reductions import TestReductions


def _reference_hash_tensor(tensor, dim=(), keepdim=False, mode=0):
    assert mode == 0, "Only mode=0 (xor_sum) is supported right now"

    tensor = tensor.cpu()
    if tensor.is_floating_point():
        tensor = tensor.to(torch.float64)
    tensor = tensor.numpy()
    dtype = tensor.dtype
    if dtype.kind == "f":
        tensor = tensor.astype(np.float64).view(np.uint64)
    else:
        tensor = tensor.astype(np.uint64)

    if dim == ():
        result = np.bitwise_xor.reduce(tensor, axis=None, keepdims=keepdim)
    else:
        if isinstance(dim, list):
            dim = tuple(dim)
        result = np.bitwise_xor.reduce(tensor, axis=dim, keepdims=keepdim)

    return result


class TestHashTensorXPU(TestCase):
    @dtypes(*all_types_and(torch.half, torch.bfloat16, torch.bool))
    def test_hash_tensor_xpu(self, device, dtype):
        # Test full reduction (dim=[])
        for shape in [(5,), (3, 4), (2, 3, 4)]:
            t = make_tensor(shape, device=device, dtype=dtype, low=-10, high=10)
            expected = _reference_hash_tensor(t)
            result = torch.hash_tensor(t)
            self.assertEqual(result.cpu().numpy(), expected)

        # Test reduction along a single dim
        t = make_tensor((3, 4), device=device, dtype=dtype, low=-10, high=10)
        for dim in [0, 1]:
            expected = _reference_hash_tensor(t, dim=dim)
            result = torch.hash_tensor(t, dim=dim)
            self.assertEqual(result.cpu().numpy(), expected)

        # Test keepdim
        t = make_tensor((3, 4), device=device, dtype=dtype, low=-10, high=10)
        expected = _reference_hash_tensor(t, dim=1, keepdim=True)
        result = torch.hash_tensor(t, dim=1, keepdim=True)
        self.assertEqual(result.cpu().numpy(), expected)

        # Test out= variant
        t = make_tensor((3, 4), device=device, dtype=dtype, low=-10, high=10)
        out = torch.empty((3,), device=device, dtype=torch.uint64)
        torch.hash_tensor(t, dim=1, out=out)
        expected = _reference_hash_tensor(t, dim=1)
        self.assertEqual(out.cpu().numpy(), expected)

    def test_hash_tensor_float_xpu(self, device):
        # Floating point inputs are hashed via their bit pattern
        t = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float32)
        expected = _reference_hash_tensor(t)
        result = torch.hash_tensor(t)
        self.assertEqual(result.cpu().numpy(), expected)

    def test_hash_tensor_empty_xpu(self, device):
        t = torch.empty((0, 3), device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "Expected reduction dim"):
            torch.hash_tensor(t)
        with self.assertRaisesRegex(IndexError, "non-zero size"):
            torch.hash_tensor(t, dim=0)
        self.assertEqual(
            torch.hash_tensor(t, dim=1).cpu(), torch.hash_tensor(t.cpu(), dim=1)
        )

    def test_hash_tensor_consistency_xpu(self, device):
        # Same input produces the same hash
        t = make_tensor((4, 4), device=device, dtype=torch.int64, low=-10, high=10)
        self.assertEqual(torch.hash_tensor(t), torch.hash_tensor(t))

    def test_hash_tensor_dispatch_xpu(self, device):
        # Verify hash_tensor.out has a native XPU kernel registered, rather
        # than silently falling back to CPU. This guards against accidentally
        # re-adding the op to the XPUFallback.cpp fallback list.
        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::hash_tensor.out", "XPU"
            )
        )


instantiate_device_type_tests(
    TestHashTensorXPU, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(TestReductions, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
