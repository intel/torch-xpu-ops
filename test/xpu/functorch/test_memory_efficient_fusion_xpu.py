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

# Owner(s): ["module: functorch"]

import inspect
import unittest
from collections.abc import Callable

import torch
import torch.nn as nn
from functorch.compile import memory_efficient_fusion
from torch.nn import functional as F
from torch.testing._internal.common_utils import run_tests, TestCase

HAS_GPU = torch.cuda.is_available() or torch.xpu.is_available()


def _num_args(fn: Callable):
    return len(inspect.signature(fn).parameters)


def gelu_bias(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x.mul(torch.tanh(F.softplus(x)))


def hard_sigmoid(x):
    return (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_swish(x):
    return x * (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_mish(x):
    return 0.5 * x * (x + 2.0).clamp(min=0.0, max=2.0)


def run_and_compare_activation(self, fn, inps):
    with torch.jit.fuser("fuser1"):
        device = "cuda" if torch.cuda.is_available() else "xpu"
        dtype = torch.float
        if isinstance(fn, nn.Module):
            fn = fn.to(device=device, dtype=dtype)

        ref_args = [
            torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            for shape in inps
        ]
        res_args = [i.detach().clone().requires_grad_(True) for i in ref_args]

        ref = fn(*ref_args)
        ref.sum().backward()

        mem_optimized_fn = memory_efficient_fusion(fn)
        for _ in range(5):
            for i in res_args:
                i.grad = None
            res = mem_optimized_fn(*res_args)
            res.sum().backward()

        self.assertEqual(ref, res)
        for ref_arg, res_arg in zip(ref_args, res_args):
            self.assertEqual(ref_arg.grad, res_arg.grad)


@unittest.skipIf(not HAS_GPU, "CUDA or XPU is unavailable")
class TestMemoryEfficientOpAuthoring(TestCase):
    def test_gelu_bias(self):
        run_and_compare_activation(self, gelu_bias, [(1024,), (1024,)])

    def test_mish(self):
        run_and_compare_activation(self, mish, [(1024,)])

    def test_swish(self):
        run_and_compare_activation(self, swish, [(1024,)])

    def test_hard_sigmoid(self):
        run_and_compare_activation(self, hard_sigmoid, [(1024,)])

    def test_hard_swish(self):
        run_and_compare_activation(self, hard_swish, [(1024,)])

    def test_layer_norm(self):
        def layer_norm(x, weight, bias):
            dim = -1
            eps = 1e-5
            mean = torch.mean(x, dim, keepdim=True)
            centered = x - mean
            var = torch.sum(centered * centered, dim, keepdim=True) / x.size(-1)
            rvar = 1.0 / torch.sqrt(var + eps)
            normed = (x - mean) * rvar
            return normed * weight + bias

        bs = 10
        ln_size = 16
        layer_norm_inps = [(bs, ln_size), (ln_size,), (ln_size,)]
        run_and_compare_activation(self, layer_norm, layer_norm_inps)

    def test_rmsnorm(self):
        class T5LayerNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)

                return self.weight * hidden_states

        bs = 256
        seq = 256
        hidden = 1024
        t5_norm = T5LayerNorm(hidden)
        t5_norm_inputs = [(bs, seq, hidden)]
        run_and_compare_activation(self, t5_norm, t5_norm_inputs)


if __name__ == "__main__":
    run_tests()
