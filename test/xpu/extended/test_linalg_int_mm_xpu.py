# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.inductor_utils import requires_triton

INT4_K_GROUP_SIZES = [
    (k, group_size)
    for k in (32, 64, 128, 256, 512, 1024)
    for group_size in (32, 64, 128)
    if k % group_size == 0
]


def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
    assert w.dim() == 2
    w = w.transpose(0, 1).contiguous()
    assert q_group_size > 1
    assert w.shape[-1] % q_group_size == 0

    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert torch.isnan(scales).sum() == 0
    zeros = min_val + scales * (2 ** (n_bit - 1))
    assert torch.isnan(zeros).sum() == 0

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    assert torch.isnan(out).sum() == 0

    out = out.to(dtype=torch.uint8).reshape(w.shape)
    # XPU int4 packing keeps the even k index in the low nibble of each byte.
    out = (out[::, 1::2] << 4 | out[::, ::2]).to(torch.uint8)

    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1)
    scales_and_zeros = torch.cat(
        [
            scales.reshape(scales.size(0), scales.size(1), 1),
            zeros.reshape(zeros.size(0), zeros.size(1), 1),
        ],
        2,
    )
    scales_and_zeros = scales_and_zeros.transpose(0, 1).contiguous()
    return out, scales_and_zeros


class TestLinalgIntMM(TestCase):
    @parametrize("k", [16, 32])
    @parametrize("n", [16, 32])
    @parametrize("use_transpose_a", [True, False])
    @parametrize("use_transpose_b", [True, False])
    def test__int_mm(self, device, k, n, use_transpose_a, use_transpose_b):
        def genf_int_float(x, y, use_transpose):
            if use_transpose:
                x, y = y, x
            x_int8 = torch.randint(-10, 10, (x, y), dtype=torch.int8, device=device)
            x_float = x_int8.to(torch.float32)
            if use_transpose:
                return x_int8.t(), x_float.t()
            return x_int8, x_float

        m = 17
        a_int8, a_float = genf_int_float(m, k, use_transpose_a)
        b_int8, b_float = genf_int_float(k, n, use_transpose_b)
        ref = torch.mm(a_float, b_float)

        c_int32 = torch._int_mm(a_int8, b_int8)
        self.assertTrue(c_int32.dtype is torch.int32)
        self.assertEqual(c_int32.device, torch.device(device))
        self.assertEqual(c_int32.float(), ref)

        c_int32_result = c_int32.new_empty(c_int32.size())
        torch._int_mm(a_int8, b_int8, out=c_int32_result)
        self.assertEqual(c_int32_result.float(), ref)

    @parametrize("m", [1, 32, 64, 1024])
    @parametrize("k,group_size", INT4_K_GROUP_SIZES)
    @parametrize("n", [32, 48, 64, 128, 256, 512, 1024])
    def test__int4_mm(self, device, m, k, group_size, n):
        inner_k_tiles = 2

        torch.manual_seed(1)
        a_bf16 = torch.rand((m, k), dtype=torch.bfloat16, device=device)
        b_bf16 = torch.rand((k, n), dtype=torch.bfloat16, device=device)

        b_tmp, b_scales_and_zeros_bf16 = _group_quantize_tensor(
            b_bf16, n_bit=4, q_group_size=group_size
        )
        b_int4pack = torch._convert_weight_to_int4pack(b_tmp, inner_k_tiles)
        self.assertTrue(b_int4pack.dtype is torch.int32)
        self.assertEqual(b_int4pack.dim(), 2)

        for dtype in (torch.bfloat16, torch.float16):
            a = a_bf16.to(dtype=dtype)
            b = b_bf16.to(dtype=dtype)
            b_scales_and_zeros = b_scales_and_zeros_bf16.to(dtype=dtype)
            ref = torch.mm(a, b)
            res = torch._weight_int4pack_mm(
                a, b_int4pack, group_size, b_scales_and_zeros
            )

            mean_err = ((res - ref).abs() / ref).mean()
            self.assertLess(mean_err.item(), 0.05)

    @requires_triton()
    @parametrize("m", [32, 64])
    @parametrize("k", [32, 64])
    @parametrize("n", [48, 64])
    def test_compile_int4_mm(self, device, m, k, n):
        q_group = 32

        torch.manual_seed(1)
        a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
        b = torch.rand((k, n), dtype=torch.bfloat16, device=device)

        b_tmp, b_scales_and_zeros = _group_quantize_tensor(
            b, n_bit=4, q_group_size=q_group
        )

        @torch.compile
        def int4_mm(a, b_tmp, b_scales_and_zeros):
            b_int4pack = b_tmp.view(torch.int32)
            return torch._weight_int4pack_mm(a, b_int4pack, q_group, b_scales_and_zeros)

        res = int4_mm(a, b_tmp, b_scales_and_zeros)
        ref = torch.mm(a, b)

        mean_err = ((res - ref).abs() / ref).mean()
        self.assertLess(mean_err.item(), 0.05)

    def test__int_mm_errors(self, device):
        def genf_int(x, y):
            return torch.empty((x, y), dtype=torch.int8, device=device)

        # XPU does not enforce the CUDA size constraints (size(0) > 16, size(1)
        # and mat2.size(1) multiples of 8), so those checks are omitted here.
        self.assertRaisesRegex(
            RuntimeError,
            r"self.size\(1\) needs to match mat2.size\(0\) but got 8 and 7",
            lambda: torch._int_mm(genf_int(17, 8), genf_int(7, 32)),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"expected scalar type Char but found Float",
            lambda: torch._int_mm(genf_int(17, 8).float(), genf_int(8, 32)),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"expected scalar type Char but found Float",
            lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32).float()),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"Expected result dtype to be of type kInt but got float",
            lambda: torch._int_mm(
                genf_int(17, 8), genf_int(8, 32), out=genf_int(16, 32).float()
            ),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"Expected result.size\(0\) to be 17 but got 15",
            lambda: torch._int_mm(
                genf_int(17, 8), genf_int(8, 32), out=genf_int(15, 32).int()
            ),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"Expected result.size\(0\) to be 17 but got 16",
            lambda: torch._int_mm(
                genf_int(17, 8), genf_int(8, 32), out=genf_int(16, 31).int()
            ),
        )


instantiate_device_type_tests(
    TestLinalgIntMM, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
