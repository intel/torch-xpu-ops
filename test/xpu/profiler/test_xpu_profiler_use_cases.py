# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

# Use-case smoke tests mirroring the Kineto Profiler User Guide examples on
# XPU. Each test profiles a trivial GEMM model and verifies the exported
# chrome trace contains the expected kernel events.

import json
import unittest

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import (
    run_tests,
    TemporaryFileName,
    TEST_XPU,
    TestCase,
)
from torch.utils._triton import has_triton


def _kernel_events_from_trace(trace_path):
    """Return kernel-category events from a saved chrome trace."""
    with open(trace_path) as f:
        data = json.load(f)
    return [e for e in data.get("traceEvents", []) if e.get("cat") == "kernel"]


def _gemm_kernels(kernels):
    """Filter kernel events to those that look like a GEMM kernel.

    Kernel naming varies by PyTorch version (e.g. ``gemm_kernel`` on PT
    2.12+, ``xe4_gemm`` on older releases), so we match case-insensitively
    on the ``gemm`` substring.
    """
    return [k for k in kernels if "gemm" in k.get("name", "").lower()]


class XpuProfilerUseCasesTest(TestCase):
    @staticmethod
    def _gemm_inputs():
        # Tiny shapes: we only care that a GEMM kernel is traced, not perf.
        M = N = K = 4
        x = torch.randn(M, K, device="xpu")
        weight = torch.randn(K, N, device="xpu")
        return x, weight

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler_xpu_quick_start_kernel_in_trace(self):
        """Profile a GEMM with CPU+XPU activities and verify the exported
        chrome trace contains at least one kernel event with positive
        duration.
        """
        x, weight = self._gemm_inputs()

        # Warm up so first-iteration setup costs stay out of the profiled region.
        for _ in range(2):
            _ = x @ weight
        torch.xpu.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
        ) as prof:
            _ = x @ weight
            prof.step()  # flush events for the current profiling iteration
            # Sync before export so async XPU kernels have finished and are recorded.
            torch.xpu.synchronize()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            kernels = _kernel_events_from_trace(fname)

        gemm_kernels = _gemm_kernels(kernels)
        self.assertGreater(
            len(gemm_kernels),
            0,
            "No GEMM kernel in chrome trace; saw kernels: "
            f"{[k.get('name') for k in kernels]}",
        )
        for k in gemm_kernels:
            self.assertGreater(
                k.get("dur", 0),
                0,
                f"GEMM kernel '{k.get('name')}' has non-positive duration",
            )

    @unittest.skipUnless(
        TEST_XPU and has_triton(),
        "test requires XPU + Triton (Inductor backend)",
    )
    def test_profiler_xpu_torch_compile(self):
        """Profile a ``torch.compile``'d (Inductor) GEMM on XPU and verify
        both an XPU kernel event and a Dynamo ``Torch-Compiled Region``
        annotation appear in the chrome trace. Without the latter the test
        would pass even when ``torch.compile`` silently bails to eager,
        because a bare matmul falls back to the same ATen ``gemm_kernel``
        kernel either way.
        """
        x, weight = self._gemm_inputs()

        def model(t):
            return t @ weight

        compiled_model = torch.compile(model)

        # Single compile thread keeps the test light.
        with torch._inductor.config.patch(compile_threads=1):
            # First call triggers (and warms up) compilation before profiling.
            compiled_model(x)
            torch.xpu.synchronize()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            ) as prof:
                compiled_model(x)
                prof.step()
                torch.xpu.synchronize()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                data = json.load(f)
            events = data.get("traceEvents", [])
            kernels = [e for e in events if e.get("cat") == "kernel"]
            compiled_regions = [
                e for e in events if "Torch-Compiled Region" in e.get("name", "")
            ]

        gemm_kernels = _gemm_kernels(kernels)
        self.assertGreater(
            len(gemm_kernels),
            0,
            "No GEMM kernel in compiled trace; saw kernels: "
            f"{[k.get('name') for k in kernels]}",
        )
        self.assertGreater(
            len(compiled_regions),
            0,
            "No 'Torch-Compiled Region' event in trace - torch.compile "
            "did not run a compiled frame (silent fallback to eager?)",
        )

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler_xpu_graph(self):
        """Profile XPUGraph capture + replay. Graph creation must happen
        inside the profile context (Level Zero limitation) and
        ``acc_events=True`` is needed to retain kernel events emitted
        outside an active scheduling window. Requires PTI >= 0.17.
        """
        x, weight = self._gemm_inputs()

        # Warm up so first-iteration setup costs stay out of the profiled region.
        for _ in range(2):
            _ = x @ weight
        torch.xpu.synchronize()

        iterations = 3
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            acc_events=True,
        ) as prof:
            g = torch.xpu.XPUGraph()
            with torch.xpu.graph(g):
                _ = x @ weight

            with torch.inference_mode():
                for _ in range(iterations):
                    g.replay()
                    prof.step()
            torch.xpu.synchronize()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            kernels = _kernel_events_from_trace(fname)

        gemm_kernels = _gemm_kernels(kernels)
        self.assertGreaterEqual(
            len(gemm_kernels),
            iterations,
            f"Expected at least {iterations} GEMM kernel events from "
            f"XPUGraph replay, got {len(gemm_kernels)}; trace kernels: "
            f"{[k.get('name') for k in kernels]} (PTI >= 0.17 and "
            "acc_events=True are required)",
        )

    @unittest.skipIf(not TEST_XPU, "test requires XPU")
    def test_profiler_xpu_scope_profiler_config(self):
        """Smoke test the scope (HW-metrics) profiler API surface. Actual
        per-kernel metric collection also requires the scope-profiler build
        patches, ``ZET_ENABLE_METRICS=1`` and
        ``sysctl dev.xe.observation_paranoid=0``; we deliberately do not
        assert on metric args so the test stays portable.
        """
        x, weight = self._gemm_inputs()

        # Warm up so first-iteration setup costs stay out of the profiled region.
        for _ in range(2):
            _ = x @ weight
        torch.xpu.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            experimental_config=_ExperimentalConfig(
                profiler_metrics=[
                    "XVE_STALL",
                    "XVE_ACTIVE",
                    "GpuCoreClocks",
                    "AvgGpuCoreFrequencyMHz",
                ],
                profiler_measure_per_kernel=True,
            ),
        ) as prof:
            _ = x @ weight
            prof.step()
            torch.xpu.synchronize()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                data = json.load(f)

        self.assertIn("traceEvents", data)
        self.assertGreater(len(data["traceEvents"]), 0)


if __name__ == "__main__":
    run_tests()
