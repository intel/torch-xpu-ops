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

import os
import sys
import unittest

import torch
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

# test/fx is not in the XPUPatchForImport default search path; add it so
# the TestCommonPass import works.
_FX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../test/fx"))
if _FX_DIR not in sys.path:
    sys.path.insert(0, _FX_DIR)

with XPUPatchForImport(False):
    import test_common_passes
    from test_fx import _enrich_profiler_traces, TestFX

# ---- TestCommonPass: rebuild with XPU added to Devices ---------------------
# test_common_passes sets Devices = ["cpu"] (+ "cuda" when available) at
# module-scope and decorates TestCommonPass with @instantiate_parametrized_tests.
# We need to append "xpu" and rebuild the parametrized test class so the
# _xpu variants are created.
import itertools

from torch.fx.graph_module import GraphModule
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)

if torch.xpu.is_available() and "xpu" not in test_common_passes.Devices:
    test_common_passes.Devices.append("xpu")


@instantiate_parametrized_tests
class TestCommonPass(TestCase):
    @parametrize(
        "common_pass,f,device",
        itertools.product(
            test_common_passes.Passes,
            test_common_passes.Test_Cases,
            test_common_passes.Devices,
        ),
        test_common_passes.name_fn,
    )
    def test_correctness(self, common_pass, f, device):
        inp = torch.randn(10, device=device)
        traced_m = torch.fx.experimental.proxy_tensor.make_fx(f)(inp)
        P = common_pass()
        res = P(traced_m)
        modified_m = res.graph_module
        self.assertIsInstance(modified_m, GraphModule)

        inp_copy = inp.clone()
        expected = f(inp)
        result = modified_m(inp_copy)
        self.assertEqual(result, expected)

    @parametrize(
        "common_pass,f,device",
        itertools.product(
            test_common_passes.Passes,
            test_common_passes.Factory_Test_Cases,
            test_common_passes.Devices,
        ),
        test_common_passes.name_fn,
    )
    def test_correctness_factory(self, common_pass, f, device):
        inp = torch.randn(10, device=device)
        traced_m = torch.fx.experimental.proxy_tensor.make_fx(f)(inp, device)
        P = common_pass()
        res = P(traced_m)
        modified_m = res.graph_module
        self.assertIsInstance(modified_m, GraphModule)

        inp_copy = inp.clone()
        expected = f(inp, device)
        result = modified_m(inp_copy, device)
        self.assertEqual(result, expected)


# ---- TestFX profiler overrides ---------------------------------------------


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
@torch.fx.experimental._config.patch("enrich_profiler_metadata", True)
def _test_profiler_stack_trace_augmentation(self):
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 16)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(16, 10)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

    model = TestModel().xpu()
    compiled_model = torch.compile(model, backend="aot_eager", fullgraph=True)

    for _ in range(3):
        _ = compiled_model(torch.randn(10, 10, device="xpu"))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
    ) as prof:
        _ = compiled_model(torch.randn(10, 10, device="xpu"))

    actual_traces = _enrich_profiler_traces(prof)

    # XPU gap tracking: exact kernel event name and trace shape differ from
    # CUDA. We only validate that stack_trace augmentation ran and that the
    # trace references the expected aten ops for the model.
    self.assertIn("aten::addmm", actual_traces)
    self.assertIn("aten::relu", actual_traces)
    self.assertIn("stack_trace=", actual_traces)


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
@torch.fx.experimental._config.patch("enrich_profiler_metadata", True)
def _test_profiler_multiple_modules(self):
    class ModelA(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class ModelB(torch.nn.Module):
        def forward(self, x):
            return x - 1

    model_a = ModelA().xpu()
    model_b = ModelB().xpu()

    compiled_a = torch.compile(model_a, backend="aot_eager", fullgraph=True)
    compiled_b = torch.compile(model_b, backend="aot_eager", fullgraph=True)

    for _ in range(3):
        _ = compiled_a(torch.randn(10, 10, device="xpu"))
        _ = compiled_b(torch.randn(1, 3, 8, 8, device="xpu"))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
    ) as prof:
        _ = compiled_a(torch.randn(10, 10, device="xpu"))
        _ = compiled_b(torch.randn(1, 3, 8, 8, device="xpu"))

    actual_traces = _enrich_profiler_traces(prof)
    self.assertIn("aten::add", actual_traces)
    self.assertIn("aten::sub", actual_traces)
    self.assertIn("stack_trace=return x + 1", actual_traces)
    self.assertIn("stack_trace=return x - 1", actual_traces)


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
@torch.fx.experimental._config.patch("enrich_profiler_metadata", True)
def _test_profiler_nested_graph_modules(self):
    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = 5

        @torch.compiler.nested_compile_region
        def forward(self, x, y):
            m = torch.mul(x, y)
            s = m.sin()
            a = s + self.c
            return a

    model = Mod().xpu()
    compiled_model = torch.compile(model, backend="aot_eager", fullgraph=True)

    for _ in range(3):
        _ = compiled_model(
            torch.randn(10, 10, device="xpu"), torch.randn(10, 10, device="xpu")
        )

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
    ) as prof:
        _ = compiled_model(
            torch.randn(10, 10, device="xpu"), torch.randn(10, 10, device="xpu")
        )

    actual_traces = _enrich_profiler_traces(prof)
    self.assertIn("aten::mul", actual_traces)
    self.assertIn("aten::sin", actual_traces)
    self.assertIn("aten::add", actual_traces)
    self.assertIn("stack_trace=m = torch.mul(x, y)", actual_traces)
    self.assertIn("stack_trace=s = m.sin()", actual_traces)
    self.assertIn("stack_trace=a = s + self.c", actual_traces)


TestFX.test_profiler_stack_trace_augmentation = _test_profiler_stack_trace_augmentation
TestFX.test_profiler_multiple_modules = _test_profiler_multiple_modules
TestFX.test_profiler_nested_graph_modules = _test_profiler_nested_graph_modules


if __name__ == "__main__":
    run_tests()
