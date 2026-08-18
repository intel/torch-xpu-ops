# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import TestCase


class TestNNMethod(TestCase):
    def test_sort_large_slice(self, device=torch.device("xpu")):
        x = torch.randn(4, 1024000, device=device)
        res1val, res1ind = torch.sort(x, stable=True)
        torch.xpu.synchronize()
        # assertIsOrdered is too slow, so just compare to cpu
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), stable=True)
        self.assertEqual(res1val, res1val_cpu.xpu())
        self.assertEqual(res1ind, res1ind_cpu.xpu())
        res1val, res1ind = torch.sort(x, descending=True, stable=True)
        torch.xpu.synchronize()
        res1val_cpu, res1ind_cpu = torch.sort(x.cpu(), descending=True, stable=True)
        self.assertEqual(res1val, res1val_cpu.xpu())
        self.assertEqual(res1ind, res1ind_cpu.xpu())

    def test_sort_large_bool(self):
        tensor_dtype = torch.bool
        value_range = 2
        a = torch.randint(value_range, (22099,)).to(dtype=tensor_dtype).xpu()
        for dim in reversed(range(a.dim())):
            sorted_cpu, indices = torch.sort(a.cpu())
            sorted, indices = torch.sort(a)
            self.assertEqual(sorted.cpu(), sorted_cpu)
            sorted, indices = a.sort()
            self.assertEqual(sorted.cpu(), sorted_cpu)
            sorted, indices = a.sort(stable=True)
            self.assertEqual(sorted.cpu(), sorted_cpu)

    @largeTensorTest("48GB", device="xpu")
    def test_topk_num_tiles_no_overflow(self):
        n = 2**31 - 1
        # k > 256 routes through topk_out_with_sort -> segmented_radix_sort
        k = 300
        data = torch.zeros((1, n), device="xpu", dtype=torch.float16)
        values, indices = torch.topk(data, k, dim=1, largest=True, sorted=False)
        self.assertEqual(values.shape, (1, k))
        self.assertEqual(indices.shape, (1, k))
        # All input values are 0.0, so every top-k value must also be 0.0.
        self.assertTrue((values == 0.0).all())
        # Indices must be valid positions within the input dimension.
        self.assertTrue((indices >= 0).all() and (indices < n).all())

    def test_topk_nan_input(self):
        num_cols = 128
        k = 6
        n_nan_cols = 10
        for dtype in (torch.float32, torch.bfloat16):
            for num_rows in (1024, 4096, 8192):
                torch.manual_seed(0)
                x = torch.randn(num_rows, num_cols, device="cpu", dtype=dtype)
                x[:, -n_nan_cols:] = float("nan")
                cpu_vals, cpu_ids = torch.topk(x, k=k, dim=-1, sorted=True)
                xpu_vals, xpu_ids = torch.topk(x.xpu(), k=k, dim=-1, sorted=True)
                torch.xpu.synchronize()
                self.assertTrue(
                    (xpu_ids >= 0).all() and (xpu_ids < num_cols).all(),
                    f"Out-of-range indices for dtype={dtype}, N={num_rows}",
                )
                self.assertEqual(cpu_vals.isnan(), xpu_vals.cpu().isnan())
                self.assertEqual(cpu_vals.nan_to_num(), xpu_vals.cpu().nan_to_num())
                # Non-NaN positions: indices must match CPU exactly
                non_nan_mask = ~cpu_vals.isnan()
                self.assertEqual(cpu_ids[non_nan_mask], xpu_ids.cpu()[non_nan_mask])
                # NaN positions: indices must be valid, unique per row, and
                # point to NaN values
                nan_mask = cpu_vals.isnan()
                nan_ids = xpu_ids[nan_mask.xpu()]
                self.assertTrue((nan_ids >= 0).all() and (nan_ids < num_cols).all())
                for r in range(num_rows):
                    row_ids = xpu_ids[r].cpu().tolist()
                    self.assertEqual(
                        len(set(row_ids)),
                        k,
                        f"Duplicate indices for dtype={dtype}, N={num_rows}, row={r}",
                    )

    def test_topk_neginf_input(self):
        num_cols = 128
        k = 6
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.full((2048, num_cols), float("-inf"), device="cpu", dtype=dtype)
            cpu_vals, cpu_ids = torch.topk(x, k=k, dim=-1, sorted=True)
            xpu_vals, xpu_ids = torch.topk(x.xpu(), k=k, dim=-1, sorted=True)
            torch.xpu.synchronize()
            self.assertEqual(cpu_vals, xpu_vals.cpu())
            self.assertTrue(
                (xpu_ids >= 0).all() and (xpu_ids < num_cols).all(),
                f"Out-of-range indices for all -inf input, dtype={dtype}",
            )
            # indices must be unique per row
            for r in range(x.shape[0]):
                ids_list = xpu_ids[r].cpu().tolist()
                self.assertEqual(
                    len(set(ids_list)),
                    k,
                    f"Duplicate indices in row {r} for dtype={dtype}: {ids_list}",
                )

    def test_topk_equal_values(self):
        num_cols = 128
        for k in (1, 6, 8):
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.ones(2048, num_cols, device="cpu", dtype=dtype)
                cpu_vals, _ = torch.topk(x, k=k, dim=-1, sorted=True)
                xpu_vals, xpu_ids = torch.topk(x.xpu(), k=k, dim=-1, sorted=True)
                torch.xpu.synchronize()
                self.assertEqual(cpu_vals, xpu_vals.cpu())
                self.assertTrue(
                    (xpu_ids >= 0).all() and (xpu_ids < num_cols).all(),
                    f"Out-of-range indices for k={k}, dtype={dtype}",
                )
                for r in range(x.shape[0]):
                    ids_list = xpu_ids[r].cpu().tolist()
                    self.assertEqual(
                        len(set(ids_list)),
                        k,
                        f"Duplicate indices in row {r} for k={k}, dtype={dtype}: {ids_list}",
                    )

    def test_topk_random_values(self):
        num_cols = 128
        for k in (1, 6, 8):
            for largest in (True, False):
                torch.manual_seed(0)
                x = torch.randn(2048, num_cols, dtype=torch.float32)
                cpu_vals, cpu_ids = torch.topk(
                    x, k=k, dim=-1, sorted=True, largest=largest
                )
                xpu_vals, xpu_ids = torch.topk(
                    x.xpu(), k=k, dim=-1, sorted=True, largest=largest
                )
                torch.xpu.synchronize()
                self.assertEqual(cpu_vals, xpu_vals.cpu())
                self.assertEqual(cpu_ids, xpu_ids.cpu())
