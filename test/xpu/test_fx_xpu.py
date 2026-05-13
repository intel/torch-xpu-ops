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
_FX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../test/fx")
)
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
    """
    Test that map_recorded_events_to_aten_ops_with_stack_trace correctly
    augments profiler events with stack traces from FX metadata registry.
    """

    # Simple test model
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

    # Compile the model
    compiled_model = torch.compile(model, backend="aot_eager", fullgraph=True)

    # Warmup
    for _ in range(3):
        _ = compiled_model(torch.randn(10, 10, device="xpu"))

    # Profile with the compiled model
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
    ) as prof:
        result = compiled_model(torch.randn(10, 10, device="xpu"))

    actual_traces = _enrich_profiler_traces(prof)

    # Handle platform-specific event names
    if torch.version.hip:
        actual_traces = "\n".join(
            line
            for line in actual_traces.split("\n")
            if "hipGetDeviceProperties" not in line
        )
        kernel_event = "hipExtModuleLaunchKernel"
        kernel_event_relu = "hipLaunchKernel"
    else:
        kernel_event = "xpuLaunchKernel"
        kernel_event_relu = "xpuLaunchKernel"
    if IS_WINDOWS:
        expected = f"""\
event=aten::t node=t stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::transpose node=t stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::as_strided node=t stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::addmm node=addmm stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::expand node=addmm stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::as_strided node=addmm stack_trace=return F.linear(input, self.weight, self.bias)
event={kernel_event} node=addmm stack_trace=return F.linear(input, self.weight, self.bias)
event={kernel_event} node=addmm stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::relu node=relu stack_trace=return F.relu(input, inplace=self.inplace)
event=aten::clamp_min node=relu stack_trace=return F.relu(input, inplace=self.inplace)
event={kernel_event_relu} node=relu stack_trace=return F.relu(input, inplace=self.inplace)
event=aten::t node=t_1 stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::transpose node=t_1 stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::as_strided node=t_1 stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::addmm node=addmm_1 stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::expand node=addmm_1 stack_trace=return F.linear(input, self.weight, self.bias)
event=aten::as_strided node=addmm_1 stack_trace=return F.linear(input, self.weight, self.bias)
event={kernel_event} node=addmm_1 stack_trace=return F.linear(input, self.weight, self.bias)
event={kernel_event} node=addmm_1 stack_trace=return F.linear(input, self.weight, self.bias)"""
    else:
        expected = f"""\
event=aten::t node=t stack_trace=x = self.linear1(x)
event=aten::transpose node=t stack_trace=x = self.linear1(x)
event=aten::as_strided node=t stack_trace=x = self.linear1(x)
event=aten::addmm node=addmm stack_trace=x = self.linear1(x)
event={kernel_event} node=addmm stack_trace=x = self.linear1(x)
event=aten::relu node=relu stack_trace=x = self.relu(x)
event=aten::clamp_min node=relu stack_trace=x = self.relu(x)
event={kernel_event_relu} node=relu stack_trace=x = self.relu(x)
event=aten::t node=t_1 stack_trace=x = self.linear2(x)
event=aten::transpose node=t_1 stack_trace=x = self.linear2(x)
event=aten::as_strided node=t_1 stack_trace=x = self.linear2(x)
event=aten::addmm node=addmm_1 stack_trace=x = self.linear2(x)
event={kernel_event} node=addmm_1 stack_trace=x = self.linear2(x)"""

    self.assertExpectedInline(actual_traces, expected)


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
@torch.fx.experimental._config.patch("enrich_profiler_metadata", True)
def _test_profiler_multiple_modules(self):
    """
    Test that multiple compiled modules under the same profiler session
    have their events correctly augmented with stack traces.
    """

    class ModelA(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class ModelB(torch.nn.Module):
        def forward(self, x):
            return x - 1

    model_a = ModelA().xpu()
    model_b = ModelB().xpu()

    # Compile both models
    compiled_a = torch.compile(model_a, backend="aot_eager", fullgraph=True)
    compiled_b = torch.compile(model_b, backend="aot_eager", fullgraph=True)

    # Warmup
    for _ in range(3):
        _ = compiled_a(torch.randn(10, 10, device="xpu"))
        _ = compiled_b(torch.randn(1, 3, 8, 8, device="xpu"))

    # Profile both models in the same session
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
    ) as prof:
        result_a = compiled_a(torch.randn(10, 10, device="xpu"))
        result_b = compiled_b(torch.randn(1, 3, 8, 8, device="xpu"))

    actual_traces = _enrich_profiler_traces(prof)
    kernel_event = "hipLaunchKernel" if torch.version.hip else "xpuLaunchKernel"
    self.assertExpectedInline(
        actual_traces,
        f"""\
event=aten::add node=add stack_trace=return x + 1
event={kernel_event} node=add stack_trace=return x + 1
event=aten::sub node=sub stack_trace=return x - 1
event={kernel_event} node=sub stack_trace=return x - 1""",
    )


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
@torch.fx.experimental._config.patch("enrich_profiler_metadata", True)
def _test_profiler_nested_graph_modules(self):
    """
    Test that nested graph modules (e.g., graph modules calling subgraphs)
    have their events correctly augmented with stack traces.
    """

    # Model with nested structure
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

    # Compile the model (this may create nested graph modules)
    compiled_model = torch.compile(model, backend="aot_eager", fullgraph=True)

    # Warmup
    for _ in range(3):
        _ = compiled_model(
            torch.randn(10, 10, device="xpu"), torch.randn(10, 10, device="xpu")
        )

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
    ) as prof:
        result = compiled_model(
            torch.randn(10, 10, device="xpu"), torch.randn(10, 10, device="xpu")
        )

    actual_traces = _enrich_profiler_traces(prof)
    kernel_event = "hipLaunchKernel" if torch.version.hip else "xpuLaunchKernel"
    self.assertExpectedInline(
        actual_traces,
        f"""\
event=aten::mul node=mul stack_trace=m = torch.mul(x, y)
event={kernel_event} node=mul stack_trace=m = torch.mul(x, y)
event=aten::sin node=sin stack_trace=s = m.sin()
event={kernel_event} node=sin stack_trace=s = m.sin()
event=aten::add node=add stack_trace=a = s + self.c
event={kernel_event} node=add stack_trace=a = s + self.c""",
    )


TestFX.test_profiler_stack_trace_augmentation = _test_profiler_stack_trace_augmentation
TestFX.test_profiler_multiple_modules = _test_profiler_multiple_modules
TestFX.test_profiler_nested_graph_modules = _test_profiler_nested_graph_modules


if __name__ == "__main__":
    run_tests()
