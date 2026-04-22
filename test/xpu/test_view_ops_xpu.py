# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_view_ops import TestOldViewOps, TestViewOps

    def is_view_of(self, base, other):
        if (
            not other._is_view()
            or other is base
            or other._base is not base
            or base.device != other.device
        ):
            return False

        # Note: only validates storage on native device types
        # because some accelerators, like XLA, do not expose storage
        if base.device.type == "cpu" or base.device.type == "xpu":
            if base.untyped_storage().data_ptr() != other.untyped_storage().data_ptr():
                return False

        return True

    def _test_ravel_xpu(self, device):
        def _test_ravel(tensors, size, nc=False):
            for src in tensors:
                # Continuous Tensor -> View
                flat = src.ravel()
                self.assertEqual(flat.shape, torch.Size([size]))
                self.assertEqual(src.view(-1), flat)
                self.assertIs(flat._base, src)
                self.assertTrue(flat.is_contiguous())

                # Non-continuous Tensor -> Copy
                if nc:
                    nc_src = src.t()
                    nc_flat = nc_src.ravel()
                    self.assertEqual(nc_flat.shape, torch.Size([size]))
                    self.assertEqual(nc_src.contiguous().view(-1), nc_flat)
                    self.assertIsNot(nc_flat._base, src)
                    self.assertTrue(nc_flat.is_contiguous())

        # Test that flatten returns 1-dim tensor when given a 0-dim tensor
        zero_dim_tensor = torch.tensor(123, device=device)
        flat0 = zero_dim_tensor.ravel()
        one_dim_tensor = torch.tensor([123], device=device)
        flat1 = zero_dim_tensor.ravel()
        nc_ones_tensor = torch.ones(10, device=device)[::2]
        flat2 = nc_ones_tensor.ravel()

        self.assertEqual(zero_dim_tensor.shape, torch.Size([]))
        self.assertEqual(flat0.shape, torch.Size([1]))
        self.assertEqual(one_dim_tensor.shape, torch.Size([1]))
        self.assertEqual(flat1.shape, torch.Size([1]))
        self.assertEqual(nc_ones_tensor.shape, torch.Size([5]))
        self.assertEqual(flat2.shape, torch.Size([5]))
        self.assertEqual(flat0, one_dim_tensor)
        self.assertEqual(flat0, flat1)
        self.assertEqual(flat0.shape, flat1.shape)
        self.assertTrue(flat0.is_contiguous())
        self.assertTrue(flat1.is_contiguous())
        self.assertTrue(flat2.is_contiguous())

        # Test both float tensor and quantized tensor
        tensors = [
            torch.randn(5, 5, 5, 5, device=device),
        ]
        _test_ravel(tensors, 625)

        tensors = [
            torch.randn(0, 2, 3, device=device),
            torch.randn(3, 0, 2, device=device),
        ]
        _test_ravel(tensors, 0)

        tensors = [
            torch.randn(5, 5, device=device),
        ]
        _test_ravel(tensors, 25, True)

    def _test_flatten_xpu(self, device):
        # Test that flatten returns 1-dim tensor when given a 0-dim tensor
        zero_dim_tensor = torch.tensor(123, device=device)
        flat0 = zero_dim_tensor.flatten()
        one_dim_tensor = torch.tensor([123], device=device)
        flat1 = zero_dim_tensor.flatten()

        self.assertEqual(zero_dim_tensor.shape, torch.Size([]))
        self.assertEqual(flat0.shape, torch.Size([1]))
        self.assertEqual(one_dim_tensor.shape, torch.Size([1]))
        self.assertEqual(flat1.shape, torch.Size([1]))
        self.assertEqual(flat0, one_dim_tensor)
        self.assertEqual(flat0, flat1)
        self.assertEqual(flat0.shape, flat1.shape)

        # Test both float tensor and quantized tensor
        tensors = [torch.randn(5, 5, 5, 5, device=device)]
        for src in tensors:
            flat = src.flatten(0, -1)
            self.assertEqual(flat.shape, torch.Size([625]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(0, 2)
            self.assertEqual(flat.shape, torch.Size([125, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(0, 1)
            self.assertEqual(flat.shape, torch.Size([25, 5, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(1, 2)
            self.assertEqual(flat.shape, torch.Size([5, 25, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(2, 3)
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(-2, -1)
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(2, 2)
            self.assertEqual(flat, src)

            # out of bounds index
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                src.flatten(5, 10)

            # invalid start and end
            with self.assertRaisesRegex(
                RuntimeError, "start_dim cannot come after end_dim"
            ):
                src.flatten(2, 0)

    TestViewOps.is_view_of = is_view_of
    TestOldViewOps.test_flatten = _test_flatten_xpu
    TestOldViewOps.test_ravel = _test_ravel_xpu


instantiate_device_type_tests(
    TestViewOps, globals(), include_lazy=True, only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(TestOldViewOps, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
