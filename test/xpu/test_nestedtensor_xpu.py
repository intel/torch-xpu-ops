# Copyright 2020-2025 Intel Corporation
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

import ast
import sys
import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_ATTENTION
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    skipMeta,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
)

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_nestedtensor import (
        convert_jagged_to_nested_tensor,
        get_tolerances,
        random_nt,
        random_nt_noncontiguous_pair,
        TestNestedTensor,
        TestNestedTensorAutograd,
        TestNestedTensorDeviceType,
        TestNestedTensorOpInfo,
        TestNestedTensorSubclass,
    )

    def _test_to(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))

        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(
                t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True)
            )

            devices = [t.device]
            if t.device.type == "xpu":
                if t.device.index == -1:
                    devices.append(f"xpu:{torch.xpu.current_device()}")
                elif t.device.index == torch.xpu.current_device():
                    devices.append("xpu")
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(
                    t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True)
                )

        test_copy_behavior(nt)
        self.assertEqual(nt.device, nt.to("cpu").device)
        self.assertEqual(nt.device, nt.to("cpu", dtype=torch.float32).device)
        self.assertIs(torch.float32, nt.to("cpu", dtype=torch.float32).dtype)
        self.assertEqual(nt.device, nt.to(torch.float32).device)
        self.assertIs(torch.float32, nt.to(dtype=torch.float32).dtype)

        def test_data_ptr(getter):
            self.assertEqual(getter(nt), getter(nt.to("cpu")))
            self.assertEqual(
                getter(nt), getter(nt.to(dtype=nt.dtype, device=nt.device, copy=False))
            )
            self.assertEqual(getter(nt), getter(nt.to("cpu", copy=False)))
            self.assertNotEqual(getter(nt), getter(nt.to("cpu", copy=True)))

        test_data_ptr(lambda nt: nt.data_ptr())

        if torch.xpu.is_available():
            for non_blocking in [True, False]:
                for xpu in [
                    "xpu",
                    "xpu:0" if torch.xpu.device_count() == 1 else "xpu:1",
                ]:
                    nt2 = random_nt(xpu, torch.float32, ntensors, (4, 4))
                    test_copy_behavior(nt2, non_blocking)
                    self.assertEqual(
                        nt2.device, nt2.to(xpu, non_blocking=non_blocking).device
                    )
                    self.assertEqual(
                        nt.device, nt2.to("cpu", non_blocking=non_blocking).device
                    )
                    self.assertEqual(
                        nt2.device, nt.to(xpu, non_blocking=non_blocking).device
                    )
                    self.assertIs(
                        torch.int32,
                        nt2.to(
                            "cpu", dtype=torch.int32, non_blocking=non_blocking
                        ).dtype,
                    )
                    self.assertEqual(
                        nt.device,
                        nt2.to(
                            "cpu", dtype=torch.int32, non_blocking=non_blocking
                        ).device,
                    )
                    self.assertIs(torch.int32, nt2.to(dtype=torch.int32).dtype)
                    self.assertEqual(nt2.device, nt2.to(dtype=torch.int32).device)

    def _test_copy_(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        nt_copy = torch.empty_like(nt)
        nt_copy.copy_(nt)

        for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
            self.assertEqual(nt_ub, nt_copy_ub)

        nt_error = torch.nested.nested_tensor([torch.tensor([0, 0])])
        self.assertRaisesRegex(
            RuntimeError,
            "copy_ only supports tensors that are the same size for Nested implementations",
            lambda: nt_error.copy_(nt),
        )

        if torch.xpu.is_available():
            nt = random_nt(torch.device("xpu"), torch.float32, ntensors, (4, 4))
            nt_copy = torch.empty_like(nt, device=torch.device("cpu"))
            nt_copy.copy_(nt, non_blocking=True)
            torch.xpu.current_stream(torch.xpu.current_device()).synchronize()
            for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

            nt_copy = torch.empty_like(nt, device=torch.device("cpu"))
            nt_copy.copy_(nt, non_blocking=False)
            for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

    @skipMeta
    def _test_device_checks(self, device):
        nt = torch.nested.nested_tensor([], device=device)
        is_xpu = "xpu" in str(device)
        self.assertEqual(nt.is_xpu, is_xpu)

    @dtypes(torch.float, torch.float16, torch.double)
    def _test_empty_like(self, device, dtype):
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))

        # Create empty on same device as original nested tensor
        nt_empty = torch.empty_like(nt)
        assert nt.is_same_size(nt_empty)
        self.assertEqual(nt.dtype, nt_empty.dtype)
        self.assertEqual(nt.device, nt_empty.device)
        self.assertEqual(nt.layout, nt_empty.layout)

        if torch.xpu.is_available():
            if device == "cpu":
                nt_xpu = torch.empty_like(nt, device="xpu")
                self.assertEqual(torch.device("xpu").type, nt_xpu.device.type)
            else:
                nt_cpu = torch.empty_like(nt, device="cpu")
                self.assertEqual(torch.device("cpu").type, nt_cpu.device.type)

        # Check changing dtype of empty_like nested tensor output
        dtype_set = {torch.float, torch.float16, torch.double}
        for other_dtype in dtype_set - {dtype}:
            nt_empty_other_dtype = torch.empty_like(nt, dtype=other_dtype)
            self.assertEqual(nt.dtype, dtype)
            self.assertEqual(nt_empty_other_dtype.dtype, other_dtype)
            self.assertEqual(nt.device, nt_empty.device)
            self.assertEqual(nt.layout, nt_empty.layout)

        # Create tensor for autograd
        nt_empty_req_grad = torch.empty_like(nt, requires_grad=True)
        self.assertEqual(nt_empty_req_grad.requires_grad, True)

        # Test noncontiguous tensor does not fail to copy
        nt_cont, nt_noncont = random_nt_noncontiguous_pair((2, 3, 6, 7))
        nt_empty = torch.empty_like(nt_cont)
        assert nt_cont.is_same_size(nt_empty)
        nt_empty_non_contig = torch.empty_like(nt_noncont)
        assert nt_noncont.is_same_size(nt_empty_non_contig)

        # Test the contiguous memory format option
        nt_empty_contig = torch.empty_like(
            nt_cont, memory_format=torch.contiguous_format
        )
        assert nt_cont.is_same_size(nt_empty_contig)
        assert nt_empty_contig.is_contiguous()

        nt_empty_non_contig = torch.empty_like(
            nt_noncont, memory_format=torch.contiguous_format
        )
        assert nt_noncont.is_same_size(nt_empty_non_contig)
        assert nt_empty_non_contig.is_contiguous()

        # Test other memory formats fail
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_cont, memory_format=torch.channels_last),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_noncont, memory_format=torch.channels_last),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_cont, memory_format=torch.channels_last_3d),
        )
        self.assertRaises(
            RuntimeError,
            lambda: torch.empty_like(nt_noncont, memory_format=torch.channels_last_3d),
        )

    @dtypes(torch.float32)
    def _test_linear_backward_memory_usage(self, device, dtype):
        # Verify that linear_backward() doesn't use more memory than it should
        # for higher dim input sizes.
        # See https://github.com/pytorch/pytorch/issues/141112
        B, D, max_seq_len = 64, 512, 100
        m = torch.nn.Linear(D, D, device=device)
        nt = torch.nested.as_nested_tensor(
            [
                torch.rand(size=[seq_len, D])
                for seq_len in torch.randint(max_seq_len, size=(B,))
            ],
            layout=torch.jagged,
            device=device,
        )

        # (B, j1, D) -> (B, j1, 1, D) for a higher dim input size
        nt = nt.unsqueeze(-2)
        # linear_backward() should not explode the max memory usage
        torch.xpu.reset_max_memory_allocated()
        m(nt).sum().backward()
        # expect under a GB for max memory allocated
        max_after_gb = torch.xpu.max_memory_allocated(0) // (1024**3)
        self.assertEqual(max_after_gb, 0)

    @dtypes(torch.float32)
    def _test_record_stream(self, device, dtype):
        def _create_nt():
            values = torch.ones(1024, 4 * 1024, device="xpu")
            offsets = torch.tensor([0, 500, 1024], device="xpu", dtype=torch.int64)
            lengths = offsets.diff()
            nt = torch.nested.nested_tensor_from_jagged(values, offsets, lengths)
            data_ptrs = {
                nt._values.data_ptr(),
                nt._offsets.data_ptr(),
                nt._lengths.data_ptr(),
            }
            return nt, data_ptrs

        def fn(record_stream):
            nt, data_ptrs = _create_nt()
            s = torch.xpu.Stream()

            with torch.xpu.stream(s):
                # emulate doing something long via sleep
                per_ms = 2e7
                torch.xpu._sleep(int(per_ms * 100))
                if record_stream:
                    nt.record_stream(s)
            return data_ptrs

        # expect memory reuse when record_stream() is not run
        data_ptrs = fn(record_stream=False)
        nt, nt_data_ptrs = _create_nt()
        self.assertEqual(data_ptrs, nt_data_ptrs)
        del nt
        torch.xpu.synchronize()

        # expect memory to be preserved (no reuse) when record_stream() is run
        data_ptrs = fn(record_stream=True)
        nt, nt_data_ptrs = _create_nt()
        self.assertEqual(len(data_ptrs.intersection(nt_data_ptrs)), 0)

    @dtypes(torch.float32)
    def _test_construction_from_list(self, device, dtype):
        from torch.fx.experimental.symbolic_shapes import is_nested_int

        # success case: single ragged dim anywhere but the batch dim
        for nt_dim in [2, 3, 4]:
            for ragged_dim in range(1, nt_dim):
                B = 6
                shapes = [list(range(3, 3 + nt_dim - 1)) for _ in range(B)]
                for b in range(B):
                    # subtract 1 to convert to component dim space
                    shapes[b][ragged_dim - 1] = torch.randint(
                        2, 9, (1,), device=device, dtype=torch.int64
                    ).item()

                components = [
                    torch.randn(shape, device=device, dtype=dtype) for shape in shapes
                ]
                nt = torch.nested.nested_tensor(components, layout=torch.jagged)

                self.assertEqual(nt.dim(), nt_dim)
                self.assertEqual(nt._ragged_idx, ragged_dim)
                for d in range(nt_dim):
                    self.assertEqual(d == ragged_dim, is_nested_int(nt.shape[d]))

        # error case: empty list
        with self.assertRaisesRegex(
            RuntimeError, "Cannot construct a nested tensor from an empty tensor list"
        ):
            torch.nested.nested_tensor([], layout=torch.jagged)

        # error case: list of zero-dim tensors
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot construct a nested tensor from a list of zero-dim tensors",
        ):
            torch.nested.nested_tensor(
                [
                    torch.tensor(3.0, device=device, dtype=dtype),
                    torch.tensor(4.0, device=device, dtype=dtype),
                    torch.tensor(5.0, device=device, dtype=dtype),
                ],
                layout=torch.jagged,
            )

        # error case: multiple ragged dims
        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot represent given tensor list as a nested tensor with the jagged layout",
        ):
            torch.nested.nested_tensor(
                [
                    torch.randn(2, 3, device=device, dtype=dtype),
                    torch.randn(4, 5, device=device, dtype=dtype),
                ],
                layout=torch.jagged,
            )

        # error case: components on multiple devices
        if "xpu" in device:
            with self.assertRaisesRegex(
                RuntimeError,
                "When constructing a nested tensor, all tensors in list must be on the same device",
            ):
                torch.nested.nested_tensor(
                    [
                        torch.randn(2, 3, device=device, dtype=dtype),
                        torch.randn(2, 4, device="cpu", dtype=dtype),
                    ],
                    layout=torch.jagged,
                )

        # error case: components with multiple dtypes
        with self.assertRaisesRegex(
            RuntimeError,
            "When constructing a nested tensor, all tensors in list must have the same dtype",
        ):
            torch.nested.nested_tensor(
                [
                    torch.randn(2, 3, device=device, dtype=dtype),
                    torch.randn(2, 4, device=device, dtype=torch.float64),
                ],
                layout=torch.jagged,
            )

        # error case: components with multiple dims
        with self.assertRaisesRegex(
            RuntimeError,
            "When constructing a nested tensor, all tensors in list must have the same dim",
        ):
            torch.nested.nested_tensor(
                [
                    torch.randn(2, 3, device=device, dtype=dtype),
                    torch.randn(2, 3, 4, device=device, dtype=dtype),
                ],
                layout=torch.jagged,
            )

    def _test_index_put_error(self, device):
        import subprocess

        with self.subTest():
            r = subprocess.call(
                [
                    sys.executable,
                    "-c",
                    """\
import torch
offsets = torch.tensor([0, 2, 5, 7], device='xpu')
lengths = torch.tensor([2, 2, 2], device='xpu')
indices = [
    torch.tensor([0, 1, 2], device='xpu'),
    torch.tensor([0, 2, 1], device='xpu'),
    torch.tensor([0, 0, 0], device='xpu'),
]
a = torch.nested.nested_tensor_from_jagged(
    torch.zeros(7, 3, device='xpu'), offsets, lengths
)
a[indices] = 1.0
torch.xpu.synchronize()
""",
                ]
            )
            self.assertTrue(r != 0)

    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    def _test_sdpa(self, device, dtype):
        batch_size = 1
        emb_dims = 128
        n_heads = 8
        head_dims = emb_dims // n_heads

        sen1 = torch.randn(11, emb_dims, dtype=dtype, device=device)
        sen2 = torch.randn(13, emb_dims, dtype=dtype, device=device)

        query = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )
        key = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )
        value = torch.nn.Linear(
            emb_dims, emb_dims, bias=False, device=device, dtype=dtype
        )

        # Simplest case: 1 sentence, no batching
        x_d1 = sen1.unsqueeze(0)
        x_nt = torch.nested.as_nested_tensor([sen1], layout=torch.jagged)

        # See note below for why we detach here.
        q_d1 = (
            query(x_d1)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        q_d1_t = q_d1.transpose(1, 2)
        k_d1 = (
            key(x_d1)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        k_d1_t = k_d1.transpose(1, 2)
        v_d1 = (
            value(x_d1)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        v_d1_t = v_d1.transpose(1, 2)

        q_nt = (
            query(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        q_nt_t = q_nt.transpose(1, 2)
        k_nt = (
            key(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        k_nt_t = k_nt.transpose(1, 2)
        v_nt = (
            value(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        v_nt_t = v_nt.transpose(1, 2)

        # High Precision Math Reference
        q_d1_f32 = q_d1.to(torch.float32)
        k_d1_f32 = k_d1.to(torch.float32)
        v_d1_f32 = v_d1.to(torch.float32)
        q_d1_f32_t = q_d1_f32.transpose(1, 2)
        k_d1_f32_t = k_d1_f32.transpose(1, 2)
        v_d1_f32_t = v_d1_f32.transpose(1, 2)
        out_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q_d1_f32_t, k_d1_f32_t, v_d1_f32_t
        )[0]
        grads_ref = torch.autograd.grad(out_ref.sum(), (q_d1_f32, k_d1_f32, v_d1_f32))

        # Low Precision Math Reference
        out_lp_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q_d1_t, k_d1_t, v_d1_t
        )[0]
        grads_lp_ref = torch.autograd.grad(out_lp_ref.sum(), (q_d1, k_d1, v_d1))

        # Compute tolerances
        output_ref_atol, output_ref_rtol = get_tolerances(out_ref, out_lp_ref)
        # fudge factor of 1.7 for smaller GPUs e.g., A2, A16
        grad_q_ref_atol, grad_q_ref_rtol = get_tolerances(
            grads_ref[0], grads_lp_ref[0], 1.7
        )
        grad_k_ref_atol, grad_k_ref_rtol = get_tolerances(grads_ref[1], grads_lp_ref[1])
        grad_v_ref_atol, grad_v_ref_rtol = get_tolerances(grads_ref[2], grads_lp_ref[2])
        grad_atols = [grad_q_ref_atol, grad_k_ref_atol, grad_v_ref_atol]
        grad_rtols = [grad_q_ref_rtol, grad_k_ref_rtol, grad_v_ref_rtol]

        attn_d1 = torch.nn.functional.scaled_dot_product_attention(
            q_d1_t, k_d1_t, v_d1_t
        ).transpose(1, 2)
        attn_nt = torch.nn.functional.scaled_dot_product_attention(
            q_nt_t, k_nt_t, v_nt_t
        ).transpose(1, 2)

        self.assertEqual(
            attn_d1,
            attn_nt.unbind()[0].unsqueeze(0),
            atol=output_ref_atol,
            rtol=output_ref_rtol,
        )

        # Simple case: 2 sentences, no extra params
        x_d2 = sen2.unsqueeze(0)
        x_nt = torch.nested.as_nested_tensor([sen1, sen2], layout=torch.jagged)

        # NB: we make sure the leaf tensor we compute gradients for is the view-ed tensor before
        # it is transposed. This is because today we cannot backward through view or unbind a
        # transposed tensor.
        q_d2 = (
            query(x_d2)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        q_d2_t = q_d2.transpose(1, 2)
        k_d2 = (
            key(x_d2)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        k_d2_t = k_d2.transpose(1, 2)
        v_d2 = (
            value(x_d2)
            .view(batch_size, -1, n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        v_d2_t = v_d2.transpose(1, 2)

        q_nt = (
            query(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        q_nt_t = q_nt.transpose(1, 2)
        k_nt = (
            key(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        k_nt_t = k_nt.transpose(1, 2)
        v_nt = (
            value(x_nt)
            .view(*x_nt.size()[0:2], n_heads, head_dims)
            .detach()
            .requires_grad_(True)
        )
        v_nt_t = v_nt.transpose(1, 2)

        attn_d2 = torch.nn.functional.scaled_dot_product_attention(
            q_d2_t, k_d2_t, v_d2_t
        ).transpose(1, 2)
        d1_grads = torch.autograd.grad(attn_d1.sum(), (q_d1, k_d1, v_d1))
        d2_grads = torch.autograd.grad(attn_d2.sum(), (q_d2, k_d2, v_d2))

        # Simple case 3: batch_size = 1, seq_len = 1
        q_3 = torch.randn(1, 8, 16, dtype=dtype, device=device)
        q_nt_3 = torch.nested.as_nested_tensor([q_3], layout=torch.jagged)
        q_nt_3 = q_nt_3.transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_nt_3, q_nt_3, q_nt_3
        )
        self.assertEqual(attn_out.shape, q_nt_3.shape)

        def check_forward_backward():
            attn_nt = torch.nn.functional.scaled_dot_product_attention(
                q_nt_t, k_nt_t, v_nt_t
            ).transpose(1, 2)

            attn_nts = attn_nt.unbind()
            self.assertEqual(
                attn_d1,
                attn_nts[0].unsqueeze(0),
                atol=output_ref_atol,
                rtol=output_ref_rtol,
            )
            self.assertEqual(
                attn_d2,
                attn_nts[1].unsqueeze(0),
                atol=output_ref_atol,
                rtol=output_ref_rtol,
            )

            nt_grads = torch.autograd.grad(attn_nt.values().sum(), (q_nt, k_nt, v_nt))
            for nt_grad, d1_grad, d2_grad, grad_atol, grad_rtol in zip(
                nt_grads, d1_grads, d2_grads, grad_atols, grad_rtols
            ):
                unbound_nt_grads = nt_grad.unbind()
                self.assertEqual(
                    d1_grad,
                    unbound_nt_grads[0].unsqueeze(0),
                    atol=grad_atol,
                    rtol=grad_rtol,
                )
                self.assertEqual(
                    d2_grad,
                    unbound_nt_grads[1].unsqueeze(0),
                    atol=grad_atol,
                    rtol=grad_rtol,
                )

        # Default
        check_forward_backward()

        # Test dispatcher works by calling only mem-effn and math (as they are safe for all devices)
        with torch.backends.xpu.sdp_kernel(
            enable_flash=False, enable_mem_efficient=True, enable_math=True
        ):
            check_forward_backward()

        # Test math fallback
        with torch.backends.xpu.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            # Math fallback doesn't work with bfloat16 on xpu because
            # "group_gemm_dispatch" not implemented for 'BFloat16'
            if not (str(device).startswith("xpu") and dtype == torch.bfloat16):
                check_forward_backward()

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FUSED_ATTENTION,
        "Platform doesn't support flash or mem-efficient attention",
    )
    @skipIfTorchDynamo()
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    def _test_sdpa_autocast(self, device):
        def fn_nt(values32, values16, offsets):
            nt32 = convert_jagged_to_nested_tensor(values32, offsets, max_length=16)
            nt16 = convert_jagged_to_nested_tensor(values16, offsets, max_length=16)
            nt32 = nt32.transpose(1, 2)
            nt16 = nt16.transpose(1, 2)
            return F.scaled_dot_product_attention(nt32, nt16, nt32)

        def fn_dense(x32, x16):
            x32 = x32.view(8, 16, 4, 16).transpose(1, 2)
            x16 = x16.view(8, 16, 4, 16).transpose(1, 2)
            return F.scaled_dot_product_attention(x32, x16, x32)

        values32 = torch.randn((8 * 16, 4, 16), device=device, dtype=torch.float32)
        values16 = torch.randn((8 * 16, 4, 16), device=device, dtype=torch.float16)
        offsets = torch.arange(0, 8 * 16 + 1, 16, device=device, dtype=torch.int32)

        x32 = values32.clone()
        x16 = values16.clone()

        with torch.autocast(device_type="xpu", dtype=torch.float16):
            out_dense_eager = fn_dense(x32, x16)
            out_dense_compiled = torch.compile(fn_dense)(x32, x16)
            out_nt_eager = fn_nt(values32, values16, offsets)
            out_nt_compiled = torch.compile(fn_nt)(values32, values16, offsets)

        self.assertEqual(out_dense_eager, out_dense_compiled)
        self.assertEqual(
            out_dense_eager.transpose(1, 2),
            out_nt_eager.values().transpose(0, 1).view(8, 16, 4, 16),
        )
        self.assertEqual(
            out_dense_eager.transpose(1, 2),
            out_nt_compiled.values().transpose(0, 1).view(8, 16, 4, 16),
        )

        def get_values():
            return tuple(
                x.detach().clone().requires_grad_(True) for x in (values32, values16)
            )

        v32_dense_eager, v16_dense_eager = get_values()
        v32_dense_compile, v16_dense_compile = get_values()
        v32_nt_eager, v16_nt_eager = get_values()
        v32_nt_compile, v16_nt_compile = get_values()

        with torch.autocast(device_type="xpu", dtype=torch.float16):
            loss_dense_eager = fn_dense(v32_dense_eager, v16_dense_eager).sum()
            loss_dense_compile = torch.compile(fn_dense)(
                v32_dense_compile, v16_dense_compile
            ).sum()
            loss_nt_eager = fn_nt(v32_nt_eager, v16_nt_eager, offsets).values().sum()
            loss_nt_compile = (
                torch.compile(fn_nt)(v32_nt_compile, v16_nt_compile, offsets)
                .values()
                .sum()
            )

        loss_dense_eager.backward()
        loss_dense_compile.backward()
        loss_nt_eager.backward()
        loss_nt_compile.backward()

        self.assertEqual(v32_dense_eager.grad, v32_dense_compile.grad)
        self.assertEqual(v32_dense_eager.grad, v32_nt_eager.grad, atol=1e-4, rtol=1e-4)
        self.assertEqual(
            v32_dense_eager.grad, v32_nt_compile.grad, atol=1e-4, rtol=1e-4
        )

        self.assertEqual(v16_dense_eager.grad, v16_dense_compile.grad)
        self.assertEqual(v16_dense_eager.grad, v16_nt_eager.grad, atol=1e-5, rtol=5e-3)
        self.assertEqual(
            v16_dense_eager.grad, v16_nt_compile.grad, atol=1e-5, rtol=5e-3
        )

    # blows up due to test parametrization otherwise
    @torch._dynamo.utils.disable_cache_limit()
    @skipIfTorchDynamo("SDPA test compiles internally")
    @unittest.skipIf(IS_WINDOWS, reason="Windows not yet supported for torch.compile")
    @dtypes(torch.float32, torch.double, torch.half)
    @parametrize("nt_dim", [2, 3, 4])
    @parametrize("requires_grad", [False, True])
    def _test_to_padded_tensor_compile(self, device, dtype, nt_dim, requires_grad):
        if dtype is torch.bool and requires_grad:
            # grads not supported for bool
            return

        if nt_dim == 2:
            post_seq_len_shape = ()
        elif nt_dim == 3:
            post_seq_len_shape = (10,)
        elif nt_dim == 4:
            post_seq_len_shape = (9, 10)

        nt = torch.nested.nested_tensor(
            [
                torch.randint(2, (n, *post_seq_len_shape), device=device, dtype=dtype)
                if dtype is torch.bool
                else torch.randn(n, *post_seq_len_shape, device=device, dtype=dtype)
                for n in range(2, 9)
            ],
            layout=torch.jagged,
            requires_grad=requires_grad,
        )

        def f(x):
            return x.sin() + 1

        from torch.nested._internal.nested_tensor import nested_from_padded

        @torch.compile(fullgraph=True)
        def g(nt):
            def _g(nt):
                PADDING_VAL = 4.2
                padded = nt.to_padded_tensor(PADDING_VAL)
                padded = f(padded)
                # NB: sum_S must be specified to use the lowering for dense -> jagged
                # and get full fusion
                return nested_from_padded(
                    padded, nt.offsets(), sum_S=nt.values().shape[0]
                )

            # NB: use checkpointing to force fusion
            return torch.utils.checkpoint.checkpoint(_g, nt, use_reentrant=False)

        expected_output = f(nt)
        if requires_grad:
            expected_output.backward(torch.ones_like(expected_output))
            expected_grad = nt.grad.detach().clone()
            nt.grad = None

        from torch._inductor.utils import run_and_get_code

        compiled_output, generated_code = run_and_get_code(g, nt)
        if requires_grad:
            compiled_output.backward(torch.ones_like(compiled_output))
            compiled_grad = nt.grad.detach().clone()
            self.assertEqual(compiled_grad, expected_grad, rtol=1e-3, atol=1e-3)

        self.assertEqual(compiled_output, expected_output, rtol=1e-3, atol=1e-3)

        # === Verify that computation fusion happens. ===
        # Fallback op call -> fusion didn't happen.
        fallback_op_calls_present = any(
            "torch.ops.aten._padded_dense_to_jagged_forward.default("
            in generated_code[i]
            or "torch.ops.aten._jagged_to_padded_dense_forward.default("
            in generated_code[i]
            for i in range(len(generated_code))
        )

        # NB: Fusion isn't supported on CPU.
        self.assertEqual("xpu" in device, not fallback_op_calls_present)

        for i in range(len(generated_code)):
            # Examine buffer construction lines in the generated code to determine
            # whether fusion occurred. If fusion happens, a 3D buffer with shape
            # (B, max_seqlen, D) should never be materialized.
            buffer_constructions = [
                line.strip()
                for line in generated_code[i].split("\n")
                if "empty_strided_xpu(" in line
            ]

            buffer_dims = [
                # buffer dim == number of elements in the tensor size tuple arg
                len(ast.parse(t).body[0].value.args[0].elts)
                for t in buffer_constructions
            ]

            if "xpu" in device:
                self.assertFalse(any(d == 3 for d in buffer_dims))

    TestNestedTensor.test_to = _test_to
    TestNestedTensor.test_copy_ = _test_copy_
    TestNestedTensorDeviceType.test_device_checks = _test_device_checks
    TestNestedTensorDeviceType.test_empty_like = _test_empty_like
    TestNestedTensorSubclass.test_linear_backward_memory_usage = (
        _test_linear_backward_memory_usage
    )
    TestNestedTensorSubclass.test_record_stream = _test_record_stream
    TestNestedTensorSubclass.test_construction_from_list = _test_construction_from_list
    TestNestedTensorSubclass.test_index_put_error = _test_index_put_error
    TestNestedTensorSubclass.test_sdpa = _test_sdpa
    TestNestedTensorSubclass.test_sdpa_autocast = _test_sdpa_autocast
    TestNestedTensorSubclass.test_to_padded_tensor_compile = (
        _test_to_padded_tensor_compile
    )


instantiate_parametrized_tests(TestNestedTensor)
instantiate_device_type_tests(
    TestNestedTensorDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestNestedTensorAutograd, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestNestedTensorSubclass, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestNestedTensorOpInfo, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
