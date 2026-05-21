# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
# Regression test for https://github.com/intel/torch-xpu-ops/issues/2518
#
# PyTorch changed Tensor.as_subclass() to raise TypeError (instead of
# RuntimeError) for classes that do not inherit from Tensor. This test
# verifies the correct exception type is raised and that valid Tensor
# subclasses work as expected.
#
# Run: pytest test/repro/test_as_subclass.py

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class SubTensor(torch.Tensor):
    member_var = object()


class BadSubTensor:
    """A class that does not inherit from torch.Tensor."""

    member_var = object()


class TestAsSubclass(TestCase):
    def test_as_subclass_valid_subtype(self):
        """as_subclass with a proper Tensor subclass should work correctly."""
        t0 = torch.tensor(0)
        t1 = torch.tensor([1, 2])
        t2 = torch.tensor([[3, 4], [5, 6]])

        s0 = t0.as_subclass(SubTensor)
        s1 = t1.as_subclass(SubTensor)
        s2 = t2.as_subclass(SubTensor)

        # Correct type is returned
        self.assertIsInstance(s0, SubTensor)
        self.assertIsInstance(s1, SubTensor)
        self.assertIsInstance(s2, SubTensor)

        # Data is shared (as_subclass creates a view, not a copy)
        self.assertTrue(torch.equal(t0, s0))
        self.assertTrue(torch.equal(t1, s1))
        self.assertTrue(torch.equal(t2, s2))

        # Member variables are passed through
        self.assertIs(s0.member_var, SubTensor.member_var)

    def test_as_subclass_invalid_subtype_raises_type_error(self):
        """as_subclass raises TypeError for classes not inheriting from Tensor.

        PyTorch changed this from RuntimeError to TypeError. See:
        https://github.com/intel/torch-xpu-ops/issues/2518
        """
        t0 = torch.tensor(0)
        err_msg = (
            "Creating a Tensor subclass from a class that does not inherit from Tensor"
        )
        with self.assertRaisesRegex(TypeError, err_msg):
            t0.as_subclass(BadSubTensor)

    def test_as_subclass_data_shared_after_modification(self):
        """Modifications to the original tensor are reflected in the subclass."""
        t = torch.tensor([1, 2, 3])
        s = t.as_subclass(SubTensor)

        t[1] = 99
        self.assertTrue(torch.equal(t, s))
        self.assertEqual(s[1].item(), 99)


if __name__ == "__main__":
    run_tests()
