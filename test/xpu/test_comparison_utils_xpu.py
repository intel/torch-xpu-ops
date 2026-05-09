# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]


import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestComparisonUtils(TestCase):
    def test_all_equal_no_assert(self):
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, [1], [1], torch.float)

    def test_all_equal_no_assert_nones(self):
        t = torch.tensor([0.5])
        torch._assert_tensor_metadata(t, None, None, None)

    def test_assert_dtype(self):
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, None, torch.int32)

    def test_assert_strides(self):
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, None, [3], torch.float)

    def test_assert_sizes(self):
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, [3], [1], torch.float)

    def test_assert_device(self):
        t = torch.tensor([0.5], device="cpu")

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, device="xpu")

    def test_assert_layout(self):
        t = torch.tensor([0.5])

        with self.assertRaises(RuntimeError):
            torch._assert_tensor_metadata(t, layout=torch.sparse_coo)


if __name__ == "__main__":
    run_tests()
