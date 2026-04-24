# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


class TestLDLSolveInvalidPivots(TestCase):
    def test_ldl_solve_valid_pivots_match_cpu_on_xpu(self):
        a = torch.tensor(
            [
                [16.0, 4.0, 0.0, 0.0, 0.0],
                [4.0, 10.0, 8.0, 0.0, 0.0],
                [0.0, 8.0, 29.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 17.0, 9.0],
                [0.0, 0.0, 0.0, 9.0, 7.0],
            ]
        )
        b = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 1.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ]
        )

        ld_cpu, pivots_cpu = torch.linalg.ldl_factor(a, hermitian=False)
        expected = torch.linalg.ldl_solve(ld_cpu, pivots_cpu, b, hermitian=False)

        result = torch.linalg.ldl_solve(
            ld_cpu.to("xpu"), pivots_cpu.to("xpu"), b.to("xpu"), hermitian=False
        )
        self.assertEqual(expected, result.cpu())

    def test_ldl_solve_invalid_pivots_raise_on_xpu(self):
        ld = torch.tensor(
            [
                [16.0, 4.0, 0.0, 0.0, 0.0],
                [4.0, 10.0, 8.0, 0.0, 0.0],
                [0.0, 8.0, 29.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 17.0, 9.0],
                [0.0, 0.0, 0.0, 9.0, 7.0],
            ],
            device="xpu",
        )
        b = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 1.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ],
            device="xpu",
        )

        invalid_pivots = (
            (torch.tensor([0, 3, 5, 4, 5], device="xpu"), r"\|pivot\| >= 1"),
            (torch.tensor([6, 3, 5, 4, 5], device="xpu"), r"\|pivot\| <= LD\.size\(-2\)"),
            (torch.tensor([-6, 3, 5, 4, 5], device="xpu"), r"\|pivot\| <= LD\.size\(-2\)"),
        )

        for pivots, error in invalid_pivots:
            with self.assertRaisesRegex(RuntimeError, error):
                torch.linalg.ldl_solve(ld, pivots, b, hermitian=False)
