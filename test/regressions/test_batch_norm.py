# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
"""
Tests for batch_norm_mean_var covering the Impl::Contiguous code path.

The kernel selects Impl::Contiguous when:
  - canUse32BitIndexMath(input) is true, AND
  - input.is_contiguous() is true, AND
  - input.strides()[1] != 1  (i.e. standard contiguous NCHW / NCL layout,
    NOT the special case where stride-1 coincides with the channel dim)

The test exercises this path via torch.nn.BatchNorm{1,2,3}d in training mode,
which internally calls batch_norm_mean_var.  Results are validated against the
equivalent CPU computation.
"""

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

cpu_device = torch.device("cpu")
xpu_device = torch.device("xpu")


class TestBatchNormContiguous(TestCase):
    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _run_bn(self, module_cls, input_cpu, weight_cpu, bias_cpu):
        """Run BatchNorm on CPU and XPU, return (cpu_out, xpu_out_on_cpu)."""
        num_features = input_cpu.shape[1]
        dtype = input_cpu.dtype

        # CPU reference
        bn_cpu = module_cls(num_features).to(cpu_device).to(torch.float32)
        bn_cpu.weight = nn.Parameter(weight_cpu.float())
        bn_cpu.bias = nn.Parameter(bias_cpu.float())
        bn_cpu.train()
        with torch.no_grad():
            out_cpu = bn_cpu(input_cpu.float())

        # XPU under test
        bn_xpu = module_cls(num_features).to(xpu_device).to(dtype)
        bn_xpu.weight = nn.Parameter(weight_cpu.to(xpu_device).to(dtype))
        bn_xpu.bias = nn.Parameter(bias_cpu.to(xpu_device).to(dtype))
        bn_xpu.train()
        with torch.no_grad():
            out_xpu = bn_xpu(input_cpu.to(xpu_device).to(dtype))

        return out_cpu, out_xpu.to(cpu_device).float()

    def _assert_bn_close(self, out_cpu, out_xpu, dtype):
        if dtype == torch.bfloat16:
            self.assertEqual(out_cpu, out_xpu, atol=8e-3, rtol=8e-3)
            return
        self.assertEqual(out_cpu, out_xpu, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------
    # Impl::Contiguous — BatchNorm2d, standard NCHW (strides()[1] = H*W > 1)
    # ------------------------------------------------------------------
    def _test_bn2d_contiguous(self, dtype):
        N, C, H, W = 4, 8, 7, 7
        # strides = [C*H*W, H*W, W, 1]  →  strides[1] = H*W = 49 ≠ 1
        # ⇒ batch_norm_choose_impl selects Impl::Contiguous
        input_cpu = torch.randn(N, C, H, W, dtype=dtype)
        self.assertTrue(input_cpu.is_contiguous())
        self.assertNotEqual(input_cpu.stride(1), 1)

        weight_cpu = torch.ones(C)
        bias_cpu = torch.zeros(C)

        out_cpu, out_xpu = self._run_bn(nn.BatchNorm2d, input_cpu, weight_cpu, bias_cpu)
        self._assert_bn_close(out_cpu, out_xpu, dtype)

    def test_bn2d_contiguous_float32(self):
        self._test_bn2d_contiguous(torch.float32)

    def test_bn2d_contiguous_float16(self):
        self._test_bn2d_contiguous(torch.float16)

    def test_bn2d_contiguous_bfloat16(self):
        self._test_bn2d_contiguous(torch.bfloat16)

    # ------------------------------------------------------------------
    # Impl::Contiguous — BatchNorm1d on 3-D input (NCL), strides()[1] = L > 1
    # ------------------------------------------------------------------
    def _test_bn1d_contiguous(self, dtype):
        N, C, L = 4, 8, 16
        # strides = [C*L, L, 1]  →  strides[1] = L = 16 ≠ 1
        input_cpu = torch.randn(N, C, L, dtype=dtype)
        self.assertTrue(input_cpu.is_contiguous())
        self.assertNotEqual(input_cpu.stride(1), 1)

        weight_cpu = torch.ones(C)
        bias_cpu = torch.zeros(C)

        out_cpu, out_xpu = self._run_bn(nn.BatchNorm1d, input_cpu, weight_cpu, bias_cpu)
        self._assert_bn_close(out_cpu, out_xpu, dtype)

    def test_bn1d_3d_contiguous_float32(self):
        self._test_bn1d_contiguous(torch.float32)

    def test_bn1d_3d_contiguous_float16(self):
        self._test_bn1d_contiguous(torch.float16)

    def test_bn1d_3d_contiguous_bfloat16(self):
        self._test_bn1d_contiguous(torch.bfloat16)

    # ------------------------------------------------------------------
    # Impl::Contiguous — BatchNorm3d (N,C,D,H,W), strides()[1] = D*H*W > 1
    # ------------------------------------------------------------------
    def _test_bn3d_contiguous(self, dtype):
        N, C, D, H, W = 2, 4, 3, 5, 5
        input_cpu = torch.randn(N, C, D, H, W, dtype=dtype)
        self.assertTrue(input_cpu.is_contiguous())
        self.assertNotEqual(input_cpu.stride(1), 1)

        weight_cpu = torch.ones(C)
        bias_cpu = torch.zeros(C)

        out_cpu, out_xpu = self._run_bn(nn.BatchNorm3d, input_cpu, weight_cpu, bias_cpu)
        self._assert_bn_close(out_cpu, out_xpu, dtype)

    def test_bn3d_contiguous_float32(self):
        self._test_bn3d_contiguous(torch.float32)

    def test_bn3d_contiguous_float16(self):
        self._test_bn3d_contiguous(torch.float16)

    def test_bn3d_contiguous_bfloat16(self):
        self._test_bn3d_contiguous(torch.bfloat16)

    # ------------------------------------------------------------------
    # Impl::Contiguous input with non-contiguous running stats
    # Verify that batch_norm_update_stats correctly handles non-contiguous
    # running_mean / running_var tensors on XPU without error and produces
    # sane per-channel statistics.
    # ------------------------------------------------------------------
    def test_bn2d_noncontiguous_running_stats_update(self):
        """
        Pass non-contiguous running_mean / running_var to
        torch.ops.aten.batch_norm_update_stats for a contiguous NCHW input.
        This covers the XPU batch-norm kernel behavior when the input tensor
        follows the Impl::Contiguous layout but the statistics buffers are
        non-contiguous views. The test asserts that the call completes
        successfully, returns finite values, and produces per-channel outputs
        of the expected shape.
        """
        N, C, H, W = 4, 8, 7, 7
        dtype = torch.float32

        input_xpu = torch.randn(N, C, H, W, device=xpu_device, dtype=dtype)
        self.assertTrue(input_xpu.is_contiguous())
        self.assertNotEqual(input_xpu.stride(1), 1)

        # Make mean/var non-contiguous by slicing every other element from a
        # doubled buffer, so they are contiguous in storage terms but the
        # strided view is non-contiguous.
        mean_noncontig = torch.zeros(C * 2, device=xpu_device, dtype=dtype)[::2]
        var_noncontig = torch.ones(C * 2, device=xpu_device, dtype=dtype)[::2]
        self.assertFalse(mean_noncontig.is_contiguous())
        self.assertFalse(var_noncontig.is_contiguous())

        # batch_norm_update_stats exercises batch_norm_mean_var internally.
        # The non-contiguous running_mean / running_var stay in XPU memory;
        # we just confirm the call does not crash and returns sane values.
        save_mean, save_var = torch.ops.aten.batch_norm_update_stats(
            input_xpu,
            mean_noncontig,
            var_noncontig,
            0.1,
        )
        # save_mean should be the per-channel mean; rough sanity: finite values
        self.assertTrue(torch.all(torch.isfinite(save_mean)))
        self.assertTrue(torch.all(torch.isfinite(save_var)))
        self.assertEqual(save_mean.shape, torch.Size([C]))
        self.assertEqual(save_var.shape, torch.Size([C]))

    # ------------------------------------------------------------------
    # Impl::Contiguous with non-trivial weight / bias (mixed-type check)
    # ------------------------------------------------------------------
    def test_bn2d_contiguous_mixed_weight_bias(self):
        N, C, H, W = 2, 16, 4, 4
        input_cpu = torch.randn(N, C, H, W)
        weight_cpu = torch.randn(C).abs() + 0.5
        bias_cpu = torch.randn(C) * 0.1

        out_cpu, out_xpu = self._run_bn(nn.BatchNorm2d, input_cpu, weight_cpu, bias_cpu)
        self.assertEqual(out_cpu, out_xpu, atol=1e-3, rtol=1e-3)

    # ------------------------------------------------------------------
    # Impl::Contiguous — verify running stats are updated correctly
    # ------------------------------------------------------------------
    def test_bn2d_contiguous_running_stats(self):
        """
        After one training step the running_mean / running_var should match
        the CPU reference values (within floating-point tolerance).
        """
        N, C, H, W = 4, 8, 7, 7
        dtype = torch.float32
        momentum = 0.1

        input_cpu = torch.randn(N, C, H, W)
        weight_cpu = torch.ones(C)
        bias_cpu = torch.zeros(C)

        bn_cpu = nn.BatchNorm2d(C, momentum=momentum).to(cpu_device).float()
        bn_cpu.weight = nn.Parameter(weight_cpu.clone())
        bn_cpu.bias = nn.Parameter(bias_cpu.clone())
        bn_cpu.train()

        bn_xpu = nn.BatchNorm2d(C, momentum=momentum).to(xpu_device).to(dtype)
        bn_xpu.weight = nn.Parameter(weight_cpu.to(xpu_device).to(dtype))
        bn_xpu.bias = nn.Parameter(bias_cpu.to(xpu_device).to(dtype))
        bn_xpu.train()

        with torch.no_grad():
            bn_cpu(input_cpu.float())
            bn_xpu(input_cpu.to(xpu_device).to(dtype))

        self.assertEqual(
            bn_cpu.running_mean,
            bn_xpu.running_mean.to(cpu_device).float(),
            atol=1e-4,
            rtol=1e-4,
        )
        self.assertEqual(
            bn_cpu.running_var,
            bn_xpu.running_var.to(cpu_device).float(),
            atol=1e-4,
            rtol=1e-4,
        )
