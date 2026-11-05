# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""
Regression test for bcomplex32 support in XPU operators (issue #4084).

Tests eq/ne, lerp, flip, add, mul, sigmoid_backward, tanh_backward with bcomplex32.
Follows PR #4055 pattern for bcomplex32 reference comparison via complex64 cast.
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBComplex32(TestCase):
    """Test newly enabled bcomplex32 support for issue #4084 operators."""

    def test_eq_ne_bcomplex32(self):
        """Test eq/ne comparison with bcomplex32 on XPU."""
        x = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.bcomplex32).xpu()

        # eq with self
        result_eq = torch.eq(x, x)
        self.assertTrue(result_eq.all())
        self.assertEqual(result_eq.dtype, torch.bool)

        # ne with self
        result_ne = torch.ne(x, x)
        self.assertFalse(result_ne.any())
        self.assertEqual(result_ne.dtype, torch.bool)

    def test_flip_bcomplex32(self):
        """Test flip (tensor transformation) with bcomplex32 on XPU."""
        x = torch.tensor([1 + 2j, 3 + 4j, 5 + 6j], dtype=torch.bcomplex32).xpu()
        result = torch.flip(x, dims=[0])

        expected = torch.tensor([5 + 6j, 3 + 4j, 1 + 2j], dtype=torch.bcomplex32).xpu()
        self.assertTrue(torch.equal(result, expected))

    def test_add_bcomplex32(self):
        """Test add with bcomplex32 on XPU."""
        x = torch.tensor([1.0, 2.0], dtype=torch.bfloat16).xpu()
        y = 1 + 2j  # Complex scalar

        result = torch.add(x, y)
        self.assertEqual(result.dtype, torch.bcomplex32)

        # Verify result values via CPU reference computation
        x_cpu = torch.tensor([1.0, 2.0], dtype=torch.float32)
        expected_cpu = torch.add(x_cpu, torch.tensor(y, dtype=torch.complex64))
        result_cpu = result.cpu().to(dtype=torch.complex64)
        torch.testing.assert_close(result_cpu, expected_cpu, rtol=1e-2, atol=1e-2)

    def test_mul_bcomplex32(self):
        """Test mul with bcomplex32 on XPU."""
        x = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.bcomplex32).xpu()
        y = 2 + 0j

        result = torch.mul(x, y)
        self.assertEqual(result.dtype, torch.bcomplex32)

        # Verify result values via CPU reference computation
        x_cpu = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64)
        expected_cpu = torch.mul(x_cpu, torch.tensor(y, dtype=torch.complex64))
        result_cpu = result.cpu().to(dtype=torch.complex64)
        torch.testing.assert_close(result_cpu, expected_cpu, rtol=1e-2, atol=1e-2)

    def test_lerp_bcomplex32(self):
        """Test lerp with bcomplex32 on XPU."""
        x = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.bcomplex32).xpu()
        y = torch.tensor([3 + 3j, 4 + 4j], dtype=torch.bcomplex32).xpu()
        weight = 0.5 + 0.0j

        result = torch.lerp(x, y, weight)
        self.assertEqual(result.dtype, torch.bcomplex32)

        # Verify result is midpoint via CPU reference computation
        x_cpu = torch.tensor([1 + 1j, 2 + 2j], dtype=torch.complex64)
        y_cpu = torch.tensor([3 + 3j, 4 + 4j], dtype=torch.complex64)
        expected_cpu = torch.lerp(x_cpu, y_cpu, 0.5)
        result_cpu = result.cpu().to(dtype=torch.complex64)
        torch.testing.assert_close(result_cpu, expected_cpu, rtol=1e-2, atol=1e-2)

    def test_sigmoid_backward_bcomplex32(self):
        """Test sigmoid_backward with bcomplex32 on XPU."""
        grad_output = torch.tensor(
            [1 + 0j, 1 + 0j], dtype=torch.bcomplex32, requires_grad=False
        ).xpu()
        output = torch.tensor(
            [0.5 + 0j, 0.7 + 0j], dtype=torch.bcomplex32, requires_grad=False
        ).xpu()

        result = torch.ops.aten.sigmoid_backward.default(grad_output, output)

        expected_cpu = torch.ops.aten.sigmoid_backward.default(
            grad_output.cpu().to(dtype=torch.complex64),
            output.cpu().to(dtype=torch.complex64),
        )
        result_cpu = result.cpu().to(dtype=torch.complex64)

        self.assertEqual(result.dtype, torch.bcomplex32)
        torch.testing.assert_close(result_cpu, expected_cpu, rtol=1e-2, atol=1e-2)

    def test_tanh_backward_bcomplex32(self):
        """Test tanh_backward with bcomplex32 on XPU."""
        grad_output = torch.tensor(
            [1 + 0j, 1 + 0j], dtype=torch.bcomplex32, requires_grad=False
        ).xpu()
        output = torch.tensor(
            [0.5 + 0j, 0.7 + 0j], dtype=torch.bcomplex32, requires_grad=False
        ).xpu()

        result = torch.ops.aten.tanh_backward.default(grad_output, output)

        expected_cpu = torch.ops.aten.tanh_backward.default(
            grad_output.cpu().to(dtype=torch.complex64),
            output.cpu().to(dtype=torch.complex64),
        )
        result_cpu = result.cpu().to(dtype=torch.complex64)

        self.assertEqual(result.dtype, torch.bcomplex32)
        torch.testing.assert_close(result_cpu, expected_cpu, rtol=1e-2, atol=1e-2)

    def test_div_bcomplex32(self):
        """Test true division with bcomplex32 on XPU."""
        x = torch.tensor([4 + 2j, 6 + 3j], dtype=torch.bcomplex32).xpu()
        y = torch.tensor([2 + 0j, 3 + 0j], dtype=torch.bcomplex32).xpu()

        result = torch.div(x, y)
        self.assertEqual(result.dtype, torch.bcomplex32)

        # Verify result values via CPU reference computation
        x_cpu = torch.tensor([4 + 2j, 6 + 3j], dtype=torch.complex64)
        y_cpu = torch.tensor([2 + 0j, 3 + 0j], dtype=torch.complex64)
        expected_cpu = torch.div(x_cpu, y_cpu)
        result_cpu = result.cpu().to(dtype=torch.complex64)
        torch.testing.assert_close(result_cpu, expected_cpu, rtol=1e-2, atol=1e-2)

    def test_div_scalar_bcomplex32(self):
        """Test true division with scalar and bcomplex32 on XPU."""
        x = torch.tensor([4 + 2j, 6 + 3j], dtype=torch.bcomplex32).xpu()
        scalar = 2.0

        result = torch.div(x, scalar)
        self.assertEqual(result.dtype, torch.bcomplex32)

        # Verify result values via CPU reference computation
        x_cpu = torch.tensor([4 + 2j, 6 + 3j], dtype=torch.complex64)
        expected_cpu = torch.div(x_cpu, scalar)
        result_cpu = result.cpu().to(dtype=torch.complex64)
        torch.testing.assert_close(result_cpu, expected_cpu, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run_tests()
