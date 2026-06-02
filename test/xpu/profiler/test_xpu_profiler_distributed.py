# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

# Use-case smoke test for distributed profiling with the XCCL backend on
# XPU (case 03 of the Kineto Profiler User Guide). Uses
# ``MultiProcessTestCase`` to spawn worker ranks so the test runs under
# pytest without an external ``torchrun`` launcher.

import json
import os
import unittest

import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TemporaryFileName,
    TEST_XPU,
)


def _gemm_kernels(kernels):
    """Filter kernel events to those whose name contains 'gemm' (any case).

    Matches ``gemm_kernel`` (PT 2.12+) and ``xe4_gemm`` (older releases).
    """
    return [k for k in kernels if "gemm" in k.get("name", "").lower()]


def skip_if_lt_x_xpu(x):
    """XPU-aware analogue of ``common_distributed.skip_if_lt_x_gpu``.

    The upstream decorator only checks CUDA / HPU device counts, so on an
    XPU-only host it always exits the worker with ``multi-gpu-{x}``.
    """
    return skip_but_pass_in_sandcastle_if(
        torch.xpu.device_count() < x,
        f"at least {x} XPU devices needed",
    )


_MISSING = [
    name
    for ok, name in (
        (dist.is_available() and dist.is_xccl_available(), "XCCL c10d backend"),
        (TEST_XPU, "XPU device"),
    )
    if not ok
]


@unittest.skipUnless(
    not _MISSING,
    f"test requires {len(_MISSING)} missing component(s): {', '.join(_MISSING)}",
)
class XpuProfilerDistributedTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @skip_if_lt_x_xpu(2)
    def test_profiler_xpu_distributed(self):
        """Profile a GEMM + ``dist.all_reduce`` on XPU with the XCCL
        backend and verify the per-rank chrome trace contains a GEMM
        kernel event. Each rank pins to its own XPU device.
        """
        # FileStore-based rendezvous avoids needing a free TCP port per run.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        try:
            # Pin each rank to its own device so collectives don't share one GPU.
            torch.xpu.set_device(self.rank)

            M = N = K = 4
            weight = torch.randn(K, N, device="xpu")
            x = torch.randn(M, K, device="xpu")

            # Warm up so first-iteration setup costs stay out of the profiled region.
            for _ in range(2):
                _ = x @ weight
            torch.xpu.synchronize()

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            ) as prof:
                output = x @ weight
                dist.all_reduce(output)
                prof.step()
            torch.xpu.synchronize()

            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    data = json.load(f)
                kernels = [
                    e for e in data.get("traceEvents", []) if e.get("cat") == "kernel"
                ]
                gemm_kernels = _gemm_kernels(kernels)

            self.assertGreater(
                len(gemm_kernels),
                0,
                f"[Rank {self.rank}] No GEMM kernel in trace; saw "
                f"kernels: {[k.get('name') for k in kernels]}",
            )
        finally:
            # Always tear down so a failing rank doesn't hang the others.
            dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
