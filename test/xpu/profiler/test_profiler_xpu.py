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
# ruff: noqa: F841

import collections
import copy
import gc
import json
import math
import mmap
import os
import pickle
import random
import re
import struct
import subprocess
import sys
import threading
import time
import unittest
import warnings
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall
from torch._inductor.utils import is_big_gpu
from torch.autograd.profiler import KinetoStepTracker, profile as _profile
from torch.autograd.profiler_legacy import profile as _profile_legacy
from torch.profiler import (
    _utils,
    DeviceType,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_JETSON,
    IS_LINUX,
    IS_WINDOWS,
    IS_X86,
    parametrize,
    run_tests,
    serialTest,
    skipIfTorchDynamo,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_CUDA,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE

if TYPE_CHECKING:
    from torch.autograd.profiler_util import FunctionEvent


# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None


def _current_accelerator_device_type():
    acc = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else None
    )
    return acc.type if acc else None


def _current_accelerator_activity():
    device_type = _current_accelerator_device_type()
    return getattr(ProfilerActivity, device_type.upper(), None) if device_type else None


@unittest.skipIf(not HAS_PSUTIL, "Requires psutil to run")
@unittest.skipIf(IS_WINDOWS, "Test is flaky on Windows")
@unittest.skipIf(not torch.accelerator.is_available(), "Accelerator is required")
class TestProfilerCUDA(TestCase):
    def test_mem_leak(self):
        """Checks that there's no memory leak when using profiler with an accelerator"""
        device = _current_accelerator_device_type()
        self.assertIsNotNone(device)
        t = torch.rand(1, 1, device=device)
        p = psutil.Process()
        last_rss = collections.deque(maxlen=5)
        for _ in range(10):
            with _profile(use_device=device, use_kineto=(device == "xpu")):
                for _ in range(1024):
                    t = torch.mm(t, t)

            gc.collect()
            torch.accelerator.empty_cache()
            last_rss.append(p.memory_info().rss)

        # with CUDA events leaking the increase in memory was ~7 MB between
        # profiler invocations above
        is_increasing = all(
            last_rss[idx] > last_rss[idx - 1] for idx in range(1, len(last_rss))
        )
        max_diff = -1
        for idx in range(1, len(last_rss)):
            max_diff = max(max_diff, last_rss[idx] - last_rss[idx - 1])
        # Legacy CUDA leak checks are strict, but XPU Kineto profiling may keep
        # allocator/caching state in host RSS across iterations.
        max_allowed_growth = 100 * 1024 if device == GPU_TYPE else 20 * 1024 * 1024
        self.assertTrue(
            not (is_increasing and max_diff > max_allowed_growth),
            msg=f"memory usage is increasing, {str(last_rss)}",
        )

    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        # emit_nvtx is CUDA-only and triggers torch.cuda.synchronize() on enter.
        # For XPU, use emit_itt when available.
        if torch.get_device_module(GPU_TYPE).is_available():
            emitter = torch.autograd.profiler.emit_nvtx
        elif torch.xpu.is_available() and torch.profiler.itt.is_available():
            emitter = torch.autograd.profiler.emit_itt
        else:
            self.skipTest("requires CUDA emit_nvtx or XPU emit_itt")

        # Only testing that annotation emitter runs when record_shapes is enabled.
        with emitter(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()

    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "requires CUDA or XPU")
    def test_cudagraph_profiling_workaround(self):
        import subprocess

        # repro taken from #75504
        # Launch in a separate process to catch hanging/illegal memory errors
        # and to make sure CUPTI isn't already initialized.
        p = subprocess.check_call(
            [
                sys.executable,
                "-c",
                """
import os
import torch
from torch.profiler import ProfilerActivity, profile

def add_one(in_: torch.Tensor):
    return in_ + 1

if torch.get_device_module(GPU_TYPE).is_available():
    sample_arg = torch.zeros(10, device=GPU_TYPE).requires_grad_(True)

    # add this before cuda graphs are created
    torch.profiler._utils._init_for_cuda_graphs()

    add_one_graphed = torch.get_device_module(GPU_TYPE).graphs.make_graphed_callables(
        add_one, sample_args=(sample_arg,)
    )
    zeros = torch.zeros(10, device=GPU_TYPE)
    out = add_one_graphed(zeros)
    if out[0] != 1:
        raise AssertionError(f"Expected out[0] == 1, got {out[0]}")

    with profile(activities=[ProfilerActivity.CPU]):
        add_one_graphed(zeros)

    with profile(activities=[ProfilerActivity.CUDA]):
        add_one_graphed(zeros)
elif torch.xpu.is_available():
    sample_arg = torch.zeros(10, device="xpu").requires_grad_(True)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        out = add_one(sample_arg)

    graph.replay()
    if out[0].cpu() != 1:
        raise AssertionError(f"Expected out[0] == 1, got {out[0]}")

    with profile(activities=[ProfilerActivity.CPU]):
        graph.replay()

    with profile(activities=[ProfilerActivity.XPU]):
        graph.replay()
else:
    raise RuntimeError("Neither CUDA nor XPU is available")
""",
            ],
            universal_newlines=True,
            timeout=60,
        )

        # ^ this will throw an exception if the script fails.


@unittest.skipIf(not torch.profiler.itt.is_available(), "ITT is required")
class TestProfilerITT(TestCase):
    def test_custom_module_input_op_ids(self):
        class MyFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

            @staticmethod
            def backward(ctx, gO):
                (x,) = ctx.saved_tensors
                return x

        def custom_layer(input_ten):
            return MyFunc.apply(input_ten)

        # Only testing that emit_itt runs when
        # record_shapes option is enabled.
        with torch.autograd.profiler.emit_itt(record_shapes=True) as prof:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            s = custom_layer(z)
            q = s.sum()
            q.backward()


@instantiate_parametrized_tests
class TestProfiler(TestCase):
    @unittest.skipIf(
        TEST_WITH_CROSSREF, "crossref intercepts calls and changes the callsite."
    )
    def test_source(self):
        """Checks that source code attribution works for eager, TS and autograd mode"""
        # avoid automatic inlining
        prev_opt = torch._C._get_graph_executor_optimize()
        torch._C._set_graph_executor_optimize(False)

        @torch.jit.script
        def ts_method_2(x, y):
            return torch.matmul(x, y)

        @torch.jit.script
        def ts_method_1(x, y, z):
            a = x + z
            w = ts_method_2(x, y) + a
            return w.sum()

        class DummyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 2, kernel_size=1, stride=2, padding=3, bias=False
                )

            def forward(self, x):
                return self.conv(x)

        mod = DummyModule()

        def call_module(x):
            return mod(x)

        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            x = torch.randn(10, 10, requires_grad=True)
            y = torch.randn(10, 10, requires_grad=True)
            z = x + y
            w = ts_method_1(x, y, z)
            v = 2 * w
            v.backward()
            a = torch.randn(2, 3, 2, 2, requires_grad=True)
            b = call_module(a)
            c = b.sum()
            c.backward()

        for e in p.function_events:
            if "aten::add" in e.name or "AddBackward" in e.name:
                self.assertTrue(any("test_profiler" in entry for entry in e.stack))
                self.assertTrue(
                    any(
                        (
                            "test_source" in entry
                            or "ts_method_1" in entry
                            or "ts_method_2" in entry
                        )
                        for entry in e.stack
                    )
                )

        if kineto_available():
            with TemporaryFileName(mode="w+") as fname:
                p.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]

                def extract(pattern: str):
                    matches = [e for e in events if re.search(pattern, e["name"])]
                    self.assertEqual(
                        len(matches), 1, repr([e["name"] for e in matches])
                    )
                    return matches[0]

                module_event = extract(r"DummyModule_0")
                wrapper_event = extract(r"call_module")
                self.assertEqual(
                    module_event["args"]["Python parent id"],
                    wrapper_event["args"]["Python id"],
                )

        torch._C._set_graph_executor_optimize(prev_opt)

    @parametrize(
        "name,thread_spec",
        {
            "basic": ((False, False),),
            "multiple_preexisting": ((False, False),) * 2,
            "open_in_scope": ((True, False),),
            "close_in_scope": ((False, True),),
            "complex": (
                # Large number of background threads
                (False, False),
                (False, False),
                (False, False),
                (False, False),
                # some of which finish during profiling
                (False, True),
                (False, True),
                # And the profiled section is also multithreaded
                (True, False),
                (True, True),
            ),
        }.items(),
        name_fn=lambda name, thread_spec: name,
    )
    @serialTest()
    @parametrize("work_in_main_thread", [True, False])
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipIf(
        torch.xpu.is_available(),
        "XPU multithreaded Python source attribution is incorrect",
    )
    def test_source_multithreaded(self, name, thread_spec, work_in_main_thread):
        """Test various threading configurations.

        `thread_spec` is a Tuple[Tuple[bool, bool], ...] where each pair is a
        thread. The first bool indicates if the thread should be started under
        the profiler context and the second is if it should be joined under the
        profiler context.
        """

        timeout = 15
        num_threads = len(thread_spec) + 1  # Main thread
        start_barrier = threading.Barrier(num_threads, timeout=timeout)
        end_barrier = threading.Barrier(num_threads, timeout=timeout)

        class Task(threading.Thread):
            def __init__(self) -> None:
                self._end_gate = threading.Event()
                super().__init__(daemon=True)
                self.start()
                self.finished = False

            def run(self):
                self._run(self._end_gate)

            def release(self):
                self._end_gate.set()

            @staticmethod
            def _run(end_gate=None):
                def known_preexisting_function():
                    start_barrier.wait()

                # Fixed point that we can use to test capture of functions
                # which are already running when profiling is enabled.
                known_preexisting_function()

                model = torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                )

                def invoked_during_run():
                    pass

                invoked_during_run()

                _ = model(torch.rand(4, 10))
                end_barrier.wait()

                if end_gate is not None:
                    end_gate.wait(timeout=timeout)

        threads = {}

        def add_threads(context: bool):
            for idx, (start_under_profiler, _) in enumerate(thread_spec):
                if start_under_profiler == context:
                    if idx in threads:
                        raise AssertionError(f"Thread index {idx} already exists")
                    threads[idx] = Task()

        def join_threads(context: bool):
            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                if end_under_profiler == context:
                    threads[idx].release()

            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                t = threads[idx]
                if end_under_profiler == context:
                    t.join(timeout=timeout)

        try:
            add_threads(False)
            with torch.profiler.profile(with_stack=True) as prof:
                # Threads added while the profiler are running will not be observed
                # since there is no way to hook into Python's thread start call to
                # register the observer. These are here purely to verify safety.
                add_threads(True)

                if work_in_main_thread:
                    Task._run()
                else:
                    start_barrier.wait()
                    end_barrier.wait()

                join_threads(True)
            join_threads(False)

        finally:
            # It is very important that we clean up everything because the
            # Python tracer will detect ALL active threads. (Even orphans from
            # prior failed tests.) If we don't clean up properly we can
            # contaminate subsequent tests.
            start_barrier.abort()
            end_barrier.abort()
            for t in threads.values():
                t.release()

            for t in threads.values():
                t.join(timeout=timeout)

            for t in threads.values():
                self.assertFalse(t.is_alive())

        roots = prof.profiler.kineto_results.experimental_event_tree()
        nodes = [
            node
            for node in _utils.traverse_dfs(roots)
            if isinstance(node.extra_fields, _ExtraFields_PyCall)
        ]
        tid_counts = collections.Counter([node.start_tid for node in nodes])

        prior_threads = sum(
            not start_under_profiler for start_under_profiler, _ in thread_spec
        )
        expected_threads = prior_threads + 1
        self.assertEqual(
            len(tid_counts), expected_threads, f"{expected_threads}, {tid_counts}"
        )
        self.assertEqual(len(nodes), sum(tid_counts.values()))

        # Profiler uses uint64_t max as a placeholder until TID can be determined.
        no_tid = 2**64 - 1
        self.assertFalse(no_tid in tid_counts)

        worker_threads = prior_threads + (1 if work_in_main_thread else 0)

        observed_preexisting = [
            node.start_tid
            for node in nodes
            if "known_preexisting_function" in node.name
        ]
        self.assertEqual(len(observed_preexisting), worker_threads)
        self.assertEqual(len(observed_preexisting), len(set(observed_preexisting)))

        observed_during_run = [
            node.start_tid for node in nodes if "invoked_during_run" in node.name
        ]
        self.assertEqual(len(observed_during_run), worker_threads)
        self.assertEqual(len(observed_during_run), len(set(observed_during_run)))

    def payload(self, use_cuda=False, tensor_size=10, device_type=None):
        device = device_type or (GPU_TYPE if use_cuda else "cpu")
        x = torch.randn(tensor_size, tensor_size, device=device)
        y = torch.randn(tensor_size, tensor_size, device=device)
        z = torch.mm(x, y)
        z = z + y
        if device != "cpu":
            z = z.cpu()

    def _check_stats(self, profiler_stats):
        self.assertGreater(profiler_stats.profiling_window_duration_sec, 0)
        self.assertGreater(profiler_stats.number_of_events, 0)
        self.assertGreater(profiler_stats.profiler_prepare_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_enable_call_duration_us, 0)
        self.assertGreater(profiler_stats.profiler_disable_call_duration_us, 0)
        self.assertGreater(profiler_stats.parse_kineto_call_duration_us, 0)
        self.assertGreater(
            profiler_stats.function_events_build_tree_call_duration_us, 0
        )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_kineto(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        use_device = GPU_TYPE if use_cuda else None
        with _profile(use_device=use_device, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        # rerun to avoid initial start overhead
        with _profile(use_device=use_device, use_kineto=True) as p:
            self.payload(use_cuda=use_cuda)

        self.assertTrue("aten::mm" in str(p))

        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
            row_limit=-1,
        )
        # print(output)
        found_gemm = False
        found_memcpy = False
        found_mm = False
        for e in p.function_events:
            if "aten::mm" in e.name:
                found_mm = True
            if "gemm" in e.name.lower() or "Cijk" in e.name:
                found_gemm = True
            if "memcpy" in e.name.lower() or "__amd_rocclr_copyBuffer" in e.name:
                found_memcpy = True
        if use_cuda:
            self.assertTrue(found_gemm)
            self.assertTrue(found_memcpy)
        else:
            self.assertTrue(found_mm)
        self._check_stats(p._stats)
        # p.export_chrome_trace("/tmp/test_trace.json")

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not TEST_MULTIGPU, "Multiple GPUs needed")
    def test_kineto_multigpu(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for gpu_id in [0, 1]:
                x = torch.randn(10, 10).cuda(gpu_id)
                y = torch.randn(10, 10).cuda(gpu_id)
                z = x.matmul(y)
                torch.get_device_module(GPU_TYPE).synchronize(gpu_id)

        is_rocm = torch.version.hip is not None
        # on ROCm, Gemm shader is hipblaslt Shader, so we use UserArgs_MT to match.
        gemm_string = "userargs_mt" if is_rocm else "gemm"
        device_string = "hip" if is_rocm else GPU_TYPE

        device_indices = set()
        found_cuda = False
        for evt in prof.events():
            if gemm_string in evt.name.lower() and evt.device_type == DeviceType.CUDA:
                device_indices.add(evt.device_index)
            if device_string in evt.name.lower() and evt.device_type == DeviceType.CPU:
                found_cuda = True

        if is_rocm:
            # Note: On ROCm, device_indices (Node IDs) may start from values other than 0 (e.g. {2, 3})
            # because systems can contain additional (non-GPU) devices detected by the kernel,
            # resulting in offset indexing. Therefore, we validate the count of unique devices,
            # not their specific indices.
            self.assertEqual(len(device_indices), 2)
        else:
            # CUDA correctly reports logical device indices
            self.assertEqual(device_indices, {0, 1})

        self.assertTrue(found_cuda)
        self._check_stats(prof._stats())

    def test_memory_profiler(self):
        def run_profiler(tensor_creation_fn):
            # collecting allocs / deallocs
            with _profile(
                profile_memory=True,
                record_shapes=True,
                use_kineto=kineto_available(),
            ) as prof:
                x = None
                with record_function("test_user_scope_alloc"):
                    x = tensor_creation_fn()
                with record_function("test_user_scope_dealloc"):
                    del x
            return prof.key_averages(group_by_input_shape=True)

        def check_metrics(stats, metric, allocs=None, deallocs=None):
            stat_metrics = {}
            # print(stats)
            for stat in stats:
                stat_metrics[stat.key] = getattr(stat, metric)
            # print(stat_metrics)
            if allocs is not None:
                for alloc_fn in allocs:
                    self.assertTrue(alloc_fn in stat_metrics)
                    self.assertGreater(
                        stat_metrics[alloc_fn], 0, f"alloc_fn = {alloc_fn}"
                    )
            if deallocs is not None:
                for dealloc_fn in deallocs:
                    self.assertTrue(dealloc_fn in stat_metrics)
                    self.assertLess(
                        stat_metrics[dealloc_fn], 0, f"alloc_fn = {dealloc_fn}"
                    )

        def create_cpu_tensor():
            return torch.rand(10, 10)

        def create_cuda_tensor():
            return torch.rand(10, 10).to(device=GPU_TYPE)

        def create_xpu_tensor():
            return torch.rand(10, 10).xpu()

        def create_mkldnn_tensor():
            return torch.rand(10, 10, dtype=torch.float32).to_mkldnn()

        stats = run_profiler(create_cpu_tensor)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=[
                "aten::empty",
                "aten::rand",
                "test_user_scope_alloc",
            ],
            deallocs=[
                "test_user_scope_dealloc",
            ],
        )

        if kineto_available():
            with TemporaryFileName(mode="w+") as fname:
                with profile(profile_memory=True) as prof:
                    x = None
                    with record_function("test_user_scope_alloc"):
                        x = create_cpu_tensor()
                    with record_function("test_user_scope_dealloc"):
                        del x
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    trace = json.load(f)
                    if "traceEvents" not in trace:
                        raise AssertionError("Expected 'traceEvents' in trace")
                    events = trace["traceEvents"]
                    found_memory_events = False
                    for evt in events:
                        if "name" not in evt:
                            raise AssertionError("Expected 'name' in event")
                        if evt["name"] == "[memory]":
                            found_memory_events = True
                            if "args" not in evt:
                                raise AssertionError("Expected 'args' in memory event")
                            if "Addr" not in evt["args"]:
                                raise AssertionError("Expected 'Addr' in event args")
                            if "Device Type" not in evt["args"]:
                                raise AssertionError(
                                    "Expected 'Device Type' in event args"
                                )
                            if "Device Id" not in evt["args"]:
                                raise AssertionError(
                                    "Expected 'Device Id' in event args"
                                )
                            if "Bytes" not in evt["args"]:
                                raise AssertionError("Expected 'Bytes' in event args")

                            # Memory should be an instantaneous event.
                            if "dur" in evt["args"]:
                                raise AssertionError("Unexpected 'dur' in event args")
                            if "cat" in evt["args"]:
                                raise AssertionError("Unexpected 'cat' in event args")
                    if not found_memory_events:
                        raise AssertionError("Expected to find memory events")

        if torch.get_device_module(GPU_TYPE).is_available():
            create_cuda_tensor()
            stats = run_profiler(create_cuda_tensor)
            check_metrics(
                stats,
                "device_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ],
            )

        if torch.xpu.is_available():
            create_xpu_tensor()
            stats = run_profiler(create_xpu_tensor)
            check_metrics(
                stats,
                "device_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::to",
                    "aten::empty_strided",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "aten::rand",
                    "aten::empty",
                ],
            )

        if torch.backends.mkldnn.is_available():
            create_mkldnn_tensor()
            stats = run_profiler(create_mkldnn_tensor)
            check_metrics(
                stats,
                "cpu_memory_usage",
                allocs=[
                    "test_user_scope_alloc",
                    "aten::rand",
                    "aten::empty",
                    "aten::to_mkldnn",
                ],
                deallocs=[
                    "test_user_scope_dealloc",
                ],
            )

        # check top-level memory events
        with _profile(profile_memory=True, use_kineto=kineto_available()) as prof:
            x = torch.rand(10, 10)
            del x
            if torch.get_device_module(GPU_TYPE).is_available():
                y = torch.rand(10, 10).to(device=GPU_TYPE)
                del y
            elif torch.xpu.is_available():
                y = torch.rand(10, 10).to("xpu")
                del y
            gc.collect()
        stats = prof.key_averages(group_by_input_shape=True)
        check_metrics(
            stats,
            "cpu_memory_usage",
            allocs=["aten::rand", "aten::empty"],
            deallocs=["[memory]"],
        )
        if torch.get_device_module(GPU_TYPE).is_available():
            check_metrics(stats, "device_memory_usage", deallocs=["[memory]"])
        elif torch.xpu.is_available():
            check_metrics(stats, "device_memory_usage", deallocs=["[memory]"])

    @unittest.skipIf(
        IS_JETSON, "Jetson has a guard against OOM since host and gpu memory are shared"
    )
    def test_oom_tracing(self):
        def run_profiler(tensor_creation_fn):
            with _profile(profile_memory=True, record_shapes=True) as prof:
                with self.assertRaisesRegex(RuntimeError, ".*[tT]ried to allocate.*"):
                    x = tensor_creation_fn()
                return prof

        def create_cuda_tensor_oom():
            device = torch.device(f"{GPU_TYPE}:0")
            return torch.empty(
                1024, 1024, 1024, 1024, dtype=torch.float32, device=device
            )

        def check_trace(fname):
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                self.assertTrue("traceEvents" in trace)
                events = trace["traceEvents"]
                found_out_of_memory_events = False
                for evt in events:
                    self.assertTrue("name" in evt)
                    if evt["name"] == "[OutOfMemory]":
                        found_out_of_memory_events = True
                        self.assertTrue("args" in evt)
                        self.assertTrue("Device Type" in evt["args"])
                        self.assertTrue("Device Id" in evt["args"])
                        self.assertTrue("Bytes" in evt["args"])

                        # Memory should be an instantaneous event.
                        self.assertTrue("dur" not in evt["args"])
                        self.assertTrue("cat" not in evt["args"])
                self.assertTrue(found_out_of_memory_events)

        if torch.get_device_module(GPU_TYPE).is_available():
            with TemporaryFileName(mode="w+") as fname:
                prof = run_profiler(create_cuda_tensor_oom)
                check_trace(fname)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_module_hierarchy(self):
        class A(nn.Module):
            def my_new_method(self, x):
                return x * 3

            def forward_impl_(self, x, y):
                return self.my_new_method(x) + y

            def forward(self, x, y):
                y = y - 2
                return self.forward_impl_(x, y)

        class B(nn.Module):
            def forward(self, x):
                return x + 2

        class C(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.A0 = A()
                self.B0 = B()

            def call_b(self, x):
                return self.B0.forward(x)

            def forward(self, x, y):
                return self.A0.forward(x, y) + self.call_b(x)

        model = C()
        model = torch.jit.script(model)
        input_a = torch.rand(128, 128)
        input_b = torch.rand(128, 128)
        op_to_module_hierarchy = {}
        op_to_module_hierarchy["aten::sub"] = ["TOP(C)::forward.A0(A)::forward."]
        op_to_module_hierarchy["aten::mul"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.SELF(A)::my_new_method."
        ]
        op_to_module_hierarchy["aten::add"] = [
            "TOP(C)::forward.A0(A)::forward.SELF(A)::forward_impl_.",
            "TOP(C)::forward.SELF(C)::call_b.B0(B)::forward.",
            "TOP(C)::forward.",
        ]
        with TemporaryFileName(mode="w+") as fname:
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                with_modules=True,
            ) as prof:
                model(input_a, input_b)
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                if "traceEvents" not in trace:
                    raise AssertionError("Expected 'traceEvents' in trace")
                events = trace["traceEvents"]
                found_memory_events = False
                for evt in events:
                    if "name" not in evt:
                        raise AssertionError("Expected 'name' in event")
                    if "args" in evt:
                        op_name = evt["name"]
                        if "Module Hierarchy" in evt["args"]:
                            hierarchy = evt["args"]["Module Hierarchy"]
                            if op_name in op_to_module_hierarchy:
                                if hierarchy not in op_to_module_hierarchy[op_name]:
                                    raise AssertionError(
                                        f"Expected hierarchy '{hierarchy}' in {op_to_module_hierarchy[op_name]}"
                                    )

    def test_high_level_trace(self):
        """Checks that python side high level events are recorded."""

        class RepeatedDataset(torch.utils.data.Dataset):
            def __init__(self, N, D_in, D_out):
                self.N = N
                self.x = torch.randn(N, D_in)
                self.y = torch.randn(N, D_out)

            def __len__(self):
                return self.N

            def __getitem__(self, idx):
                return self.x, self.y

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)

            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        class CustomSGD(torch.optim.SGD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        def train():
            for data in dataloader:
                x, y = data[0], data[1]
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        N, D_in, H, D_out = 8, 10, 5, 2
        model = TwoLayerNet(D_in, H, D_out)
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        ds = RepeatedDataset(N, D_in, D_out)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

        try:
            train()
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")

        # Create multiple instances, expect each func is hooked only one time.
        # Nested wrappers(repeated patching) will make following test fail.
        optimizer_duplicate = torch.optim.SGD(model.parameters(), lr=1e-4)
        dataloader_duplicate = torch.utils.data.DataLoader(ds, batch_size=1)

        def judge(expected_event_count, prof):
            actual_event_count = {}
            for e in prof.function_events:
                if "#" in e.name:
                    key = e.name
                    if key in expected_event_count:
                        actual_event_count[key] = (
                            actual_event_count.setdefault(key, 0) + 1
                        )
            for key, count in expected_event_count.items():
                self.assertTrue(
                    (key in actual_event_count) and (count == actual_event_count[key])
                )

        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            # "+1" because the final iteration will enter __next__ but skip the loop body.
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#SGD.step": N,
            "Optimizer.zero_grad#SGD.zero_grad": N,
        }
        judge(expected_event_count, prof)

        # Test on pickle/unpickle. Expect to work in multi-processing.
        optimizer = pickle.loads(pickle.dumps(optimizer))
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        judge(expected_event_count, prof)

        # Test on customized optimizer.
        optimizer = CustomSGD(model.parameters(), lr=1e-4)
        with _profile(use_kineto=kineto_available()) as prof:
            train()
        expected_event_count = {
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#CustomSGD.step": N,
            "Optimizer.zero_grad#CustomSGD.zero_grad": N,
        }
        judge(expected_event_count, prof)

    def test_flops(self):
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        nested_tensor = torch.nested.nested_tensor(
            [torch.randn((2, 5)), torch.randn((3, 5))], layout=torch.jagged
        )
        with _profile(
            record_shapes=True, with_flops=True, use_kineto=kineto_available()
        ) as prof:
            model(inputs)
            # test that nested tensor won't cause exception during flop compute
            nested_tensor = nested_tensor + nested_tensor
        profiler_output = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10
        )
        self.assertRegex(profiler_output, "Total M?FLOPs")
        if not (
            kineto_available() and torch.get_device_module(GPU_TYPE).is_available()
        ):
            return

        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_flops=True,
        ) as kineto_profiler:
            model(inputs)
        profiler_output = kineto_profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1
        )
        self.assertRegex(profiler_output, "Total M?FLOPs")

    def test_override_time_units(self):
        US_IN_SECOND = 1000.0 * 1000.0
        US_IN_MS = 1000.0

        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        with _profile() as prof:
            model(inputs)

        profiler_output = prof.key_averages().table(time_unit="s")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_s = f"{event.cpu_time / US_IN_SECOND:.3f}s"
            cpu_time_total_str_s = f"{event.cpu_time_total / US_IN_SECOND:.3f}s"
            self.assertTrue(cpu_time_str_s in profiler_output)
            self.assertTrue(cpu_time_total_str_s in profiler_output)

        profiler_output = prof.key_averages().table(time_unit="ms")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_ms = f"{event.cpu_time / US_IN_MS:.3f}ms"
            cpu_time_total_str_ms = f"{event.cpu_time_total / US_IN_MS:.3f}ms"
            self.assertTrue(cpu_time_str_ms in profiler_output)
            self.assertTrue(cpu_time_total_str_ms in profiler_output)

        profiler_output = prof.key_averages().table(time_unit="us")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}s).*")
        self.assertNotRegex(profiler_output, r".*(\.[0-9]{3}ms).*")
        self.assertRegex(profiler_output, r".*(\.[0-9]{3}us).*")
        for event in prof.key_averages():
            cpu_time_str_us = f"{event.cpu_time:.3f}us"
            cpu_time_total_str_us = f"{event.cpu_time_total:.3f}us"
            self.assertTrue(cpu_time_str_us in profiler_output)
            self.assertTrue(cpu_time_total_str_us in profiler_output)

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    def test_kineto_profiler_api(self):
        called_num = [0]

        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with profile(activities=supported_activities()):
            self.payload(use_cuda=use_cuda)

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
                row_limit=-1,
            )
            # print(output)
            # p.export_chrome_trace("/tmp/test_trace_" + str(called_num[0]) + ".json")
            called_num[0] += 1

        initial_step = KinetoStepTracker.current_step()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=trace_handler,
        ) as p:
            for _ in range(8):
                self.payload(use_cuda=use_cuda)
                p.step()

        self.assertEqual(called_num[0], 2)
        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 8)

        # case without schedule
        with profile(activities=supported_activities()) as p:
            self.payload(use_cuda=use_cuda)
            self.payload(use_cuda=use_cuda)
        output = p.key_averages().table(
            sort_by="self_cuda_time_total" if use_cuda else "self_cpu_time_total",
            row_limit=-1,
        )
        # print(output)

        test_schedule = torch.profiler.schedule(
            skip_first=3, wait=2, warmup=1, active=4, repeat=2
        )
        test_schedule_expected_outputs = [
            # skip first 3
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # ----
            # repeat No. 1 begin
            # wait 2
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 1 end
            # ---
            # repeat No. 2 begin
            # wait 2
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 2 end
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    def test_kineto_profiler_multiple_steppers(self):
        niters = 8
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        net = SimpleNet()
        opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        opt.zero_grad()
        inputs = torch.rand(10)

        with profile(activities=supported_activities()):
            self.payload(use_cuda=use_cuda)

        def optimizer_step():
            """This simulates a step() hook in the optimizer"""
            KinetoStepTracker.increment_step("yet_another_step")

        initial_step = KinetoStepTracker.current_step()

        def run_batch():
            out = net(inputs)
            loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
            loss.backward()
            opt.step()
            # Manually call the hook. TODO: Remove this once we add the
            # profiler step hooks in the Optimizer class that will get triggered above.
            # See https://github.com/pytorch/pytorch/issues/88446
            optimizer_step()

        for _ in range(niters):
            run_batch()

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
        ) as p:
            for _ in range(niters):
                run_batch()
                p.step()

        self.assertEqual(KinetoStepTracker.current_step(), initial_step + 2 * niters)

    def test_export_stacks(self):
        with _profile(
            with_stack=True,
            use_kineto=kineto_available(),
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)
            z = z + y

        with TemporaryFileName(mode="w+") as fname:
            p.export_stacks(fname)
            with open(fname) as f:
                lines = f.readlines()
            if len(lines) <= 0:
                raise AssertionError("Empty stacks file")
            for line in lines:
                is_int = False
                try:
                    if int(line.split(" ")[-1]) <= 0:
                        raise AssertionError("Invalid stacks record")
                    is_int = True
                except ValueError:
                    pass
                if not is_int:
                    raise AssertionError("Invalid stacks record")

    def test_experimental_config_pickle(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BytesWarning)

            # Test with default values
            config = _ExperimentalConfig()
            pickled = pickle.dumps(config)
            unpickled = pickle.loads(pickled)
            self.assertIsInstance(unpickled, _ExperimentalConfig)

            # Test with non-default values
            config = _ExperimentalConfig(
                profiler_metrics=["metric1", "metric2"],
                profiler_measure_per_kernel=True,
                verbose=True,
                performance_events=["event1", "event2"],
                enable_cuda_sync_events=True,
                adjust_profiler_step=True,
                disable_external_correlation=True,
                profile_all_threads=True,
                capture_overload_names=True,
                record_python_gc_info=True,
                expose_kineto_event_metadata=True,
                custom_profiler_config="custom_config",
            )
            pickled = pickle.dumps(config)
            unpickled = pickle.loads(pickled)
            self.assertIsInstance(unpickled, _ExperimentalConfig)

            # Test deepcopy (which uses pickle internally)
            copied = copy.deepcopy(config)
            self.assertIsInstance(copied, _ExperimentalConfig)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_tensorboard_trace_handler(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with _profile(use_device=GPU_TYPE if use_cuda else None, use_kineto=True):
            self.payload(use_cuda=use_cuda)

        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname),
            ) as p:
                for _ in range(18):
                    self.payload(use_cuda=use_cuda)
                    p.step()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-4].isdigit() and int(parts[-4]) > 0,
                    "Wrong tracing file name pattern",
                )
                if parts[-3:] == ["pt", "trace", "json"]:
                    file_num += 1
            self.assertEqual(file_num, 3)

        # test case for gzip file format
        with TemporaryDirectoryName() as dname:
            p = profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if use_cuda else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dname, use_gzip=True
                ),
            )
            p.start()
            for _ in range(18):
                self.payload(use_cuda=use_cuda)
                p.step()
            p.stop()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-5].isdigit() and int(parts[-5]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-4:], ["pt", "trace", "json", "gz"])
                file_num += 1
            self.assertEqual(file_num, 3)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_profiler_metadata(self):
        t1, t2 = torch.ones(1), torch.ones(1)
        with profile() as prof:
            torch.add(t1, t2)
            prof.add_metadata("test_key1", "test_value1")
            prof.add_metadata_json("test_key2", "[1,2,3]")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
                if "test_key1" not in trace:
                    raise AssertionError("Expected 'test_key1' in trace")
                if trace["test_key1"] != "test_value1":
                    raise AssertionError(
                        f"Expected trace['test_key1'] == 'test_value1', got {trace['test_key1']}"
                    )
                if "test_key2" not in trace:
                    raise AssertionError("Expected 'test_key2' in trace")
                if trace["test_key2"] != [1, 2, 3]:
                    raise AssertionError(
                        f"Expected trace['test_key2'] == [1, 2, 3], got {trace['test_key2']}"
                    )

    def _test_profiler_tracing(self, use_kineto):
        with _profile(use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1), torch.ones(1)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # read the trace and expect valid json
            # if the JSON generated by export_chrome_trace is not valid, this will throw and fail the test.
            with open(fname) as f:
                json.load(f)

        # test empty trace
        with _profile(use_kineto=use_kineto) as prof:
            pass
        # saving an empty trace
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            if use_kineto:
                with open(fname) as f:
                    contents = json.load(f)
                    # Some builds may not have logger observer
                    # so skip if not
                    if "WARNING" in contents:
                        found_empty_warning = False
                        for warning in contents["WARNING"]:
                            if "No Valid Trace Events" in warning:
                                found_empty_warning = True
                        self.assertTrue(found_empty_warning)

        # Same test but for cuda.
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        if not use_cuda:
            return

        device = torch.device(f"{GPU_TYPE}:0")
        with _profile(use_device=GPU_TYPE, use_kineto=use_kineto) as prof:
            t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)
            torch.add(t1, t2)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            # Now validate the json
            with open(fname) as f:
                json.load(f)

    def test_profiler_tracing(self):
        self._test_profiler_tracing(False)
        if kineto_available():
            self._test_profiler_tracing(True)

    def test_profiler_op_event_args(self):
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        with _profile(record_shapes=True) as prof:
            a = torch.ones((64, 32), dtype=torch.float32)
            c = torch.cat([a, a]).sin()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                for e in op_events:
                    args = e["args"]
                    if e["name"] == "aten::ones":
                        self.assertEqual(
                            args["Input type"],
                            ["ScalarList", "Scalar", "", "", "Scalar"],
                        )
                        self.assertEqual(
                            args["Concrete Inputs"], ["[64, 32]", "6", "", "", "False"]
                        )

                    if e["name"] == "aten::cat":
                        self.assertEqual(args["Input Dims"], [[[64, 32], [64, 32]], []])
                        self.assertEqual(args["Input type"], ["TensorList", "Scalar"])

                    # check that each op has record function id
                    self.assertGreaterEqual(
                        args.get("Record function id", -1),
                        0,
                        f"Failed finding record funciont for op = {e}",
                    )

    def test_profiler_strides(self):
        torch._C._profiler._set_record_concrete_inputs_enabled_val(True)
        base_tensor = torch.randn(1024, dtype=torch.float32)
        a = base_tensor.as_strided((16, 16), (17, 1), 0)
        b = base_tensor.as_strided((16, 16), (25, 2), 272)
        with _profile(record_shapes=True) as prof:
            c = torch.add(a, b)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                ]
                for e in op_events:
                    args = e["args"]
                    if e["name"] == "aten::add":
                        self.assertEqual(args["Input Strides"], [[17, 1], [25, 2], []])

    def test_profiler_strides_without_concrete_inputs(self):
        torch._C._profiler._set_record_concrete_inputs_enabled_val(False)
        try:
            base_tensor = torch.randn(1024, dtype=torch.float32)
            a = base_tensor.as_strided((16, 16), (17, 1), 0)
            b = base_tensor.as_strided((16, 16), (25, 2), 272)
            with _profile(record_shapes=True) as prof:
                c = torch.add(a, b)

            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    j = json.load(f)
                    op_events = [
                        e for e in j["traceEvents"] if e.get("cat", "") == "cpu_op"
                    ]
                    for e in op_events:
                        args = e["args"]
                        if e["name"] == "aten::add":
                            self.assertIn("Input Strides", args)
                            self.assertEqual(
                                args["Input Strides"], [[17, 1], [25, 2], []]
                            )
        finally:
            torch._C._profiler._set_record_concrete_inputs_enabled_val(True)

    def test_profiler_fwd_bwd_link(self):
        with _profile(use_kineto=True) as prof:
            t1, t2 = (
                torch.ones(1, requires_grad=True),
                torch.ones(1, requires_grad=True),
            )
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                events = j["traceEvents"]
                ts_to_name = {}
                flow_s_to_ts = {}
                flow_f_to_ts = {}
                for e in events:
                    if e["ph"] == "X":
                        ts_to_name[e["ts"]] = e["name"]
                    if (
                        "cat" in e
                        and "name" in e
                        and e["cat"] == "fwdbwd"
                        and e["name"] == "fwdbwd"
                    ):
                        if e["ph"] == "s":
                            flow_s_to_ts[e["id"]] = e["ts"]
                        elif e["ph"] == "f":
                            flow_f_to_ts[e["id"]] = e["ts"]

                self.assertEqual(len(flow_s_to_ts), 2)
                self.assertEqual(len(flow_f_to_ts), 2)
                self.assertIn(1, flow_s_to_ts)
                self.assertIn(1, flow_f_to_ts)
                self.assertIn(2, flow_s_to_ts)
                self.assertIn(2, flow_f_to_ts)
                s_ts_1 = flow_s_to_ts[1]
                f_ts_1 = flow_f_to_ts[1]
                s_ts_2 = flow_s_to_ts[2]
                f_ts_2 = flow_f_to_ts[2]
                self.assertTrue(
                    all(ts in ts_to_name for ts in [s_ts_1, f_ts_1, s_ts_2, f_ts_2])
                )
                self.assertTrue(
                    ts_to_name[s_ts_1] == "aten::binary_cross_entropy_with_logits"
                )
                self.assertTrue(ts_to_name[s_ts_2] == "aten::add")

    def test_profiler_disable_fwd_bwd_link(self):
        try:
            torch._C._profiler._set_fwd_bwd_enabled_val(False)

            with _profile(use_kineto=True) as prof:
                t1, t2 = (
                    torch.ones(1, requires_grad=True),
                    torch.ones(1, requires_grad=True),
                )
                z = torch.add(t1, t2)
                y = torch.ones(1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                loss.backward()

            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    j = json.load(f)
                    events = j["traceEvents"]

                    for e in events:
                        self.assertNotEqual(e.get("cat", None), "fwdbwd")
        finally:
            torch._C._profiler._set_fwd_bwd_enabled_val(True)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "requires gpu")
    def test_profiler_cuda_sync_events(self):
        device_type = _current_accelerator_device_type()
        self.assertIsNotNone(device_type)
        device = torch.device(f"{device_type}:0")
        t1, t2 = torch.ones(1, device=device), torch.ones(1, device=device)

        def workload() -> None:
            torch.add(t1, t2)
            torch.accelerator.synchronize()
            torch.add(t1, t2)

        def trace_and_check(exp_config: _ExperimentalConfig | None) -> None:
            with _profile(
                use_kineto=True,
                use_device=device_type,
                experimental_config=exp_config,
            ) as prof:
                workload()

            with TemporaryFileName(mode="w+") as fname:
                # fname = "/tmp/kineto_out.json"
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    j = json.load(f)
                    cat_or_name = "cat" if torch.version.cuda else "name"
                    cats = {e.get(cat_or_name, None) for e in j["traceEvents"]}
            event_names = (
                {"cuda_sync", "hipDeviceSynchronize"}
                if device_type == GPU_TYPE
                else {"zeEventHostSynchronize", "xpu_sync", "xpu_synchronize"}
            )
            self.assertTrue(
                any(event_name in cats for event_name in event_names)
                or any(cat is not None and "sync" in str(cat).lower() for cat in cats),
                f"Expected to find sync event found = {cats}",
            )

        print("Testing enable_cuda_sync_events in _ExperimentalConfig")
        trace_and_check(exp_config=_ExperimentalConfig(enable_cuda_sync_events=True))

        if device_type == GPU_TYPE:
            print("Testing _profiler._set_cuda_sync_enabled_val()")
            try:
                torch._C._profiler._set_cuda_sync_enabled_val(True)
                trace_and_check(exp_config=None)
            finally:
                torch._C._profiler._set_cuda_sync_enabled_val(False)

    def test_profiler_type(self):
        profiler_type = torch._C._autograd._profiler_type
        ActiveProfilerType = torch._C._profiler.ActiveProfilerType
        self.assertEqual(profiler_type(), ActiveProfilerType.NONE)

        # Autograd profiler
        with _profile_legacy():
            self.assertEqual(profiler_type(), ActiveProfilerType.LEGACY)

        # Kineto profiler
        with profile():
            self.assertEqual(profiler_type(), ActiveProfilerType.KINETO)

    def test_profiler_correlation_id(self):
        """
        We expect the correlation_id to be unique across multiple invocation of the profiler,
        So we will reuse id_uniqueness_set.
        """
        id_uniqueness_set = set()
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        inputs = torch.randn(40, 16, 18, 260)
        uint32_max = 2**32 - 1
        for _ in range(5):
            with profile() as prof:
                model(inputs)
            for event in prof.profiler.kineto_results.events():
                corr_id = event.correlation_id()
                if (corr_id) and event.device_type() == DeviceType.CPU:
                    self.assertTrue(corr_id not in id_uniqueness_set)
                    id_uniqueness_set.add(corr_id)
                    self.assertTrue(corr_id < uint32_max)

    def test_nested_tensor_with_shapes(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        c = torch.randn(4, 4)
        inp = torch.nested.nested_tensor([a, b])
        with torch.profiler.profile(record_shapes=True) as prof:
            torch.nn.functional.linear(inp, c, None)
        for e in prof.events():
            if e.name in ("aten::mm", "aten::addmm"):
                # intentionally vague tests to protect against possible future changes
                # of mm to addmm or other impl, or changing internal order of args
                self.assertTrue(len(e.input_shapes) > 0)
                self.assertTrue(len(e.input_shapes[0]) > 0)

    @patch.dict(os.environ, {"KINETO_USE_DAEMON": "1"})
    @patch.dict(os.environ, {"KINETO_DAEMON_INIT_DELAY_S": "1"})
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_kineto_profiler_with_environment_variable(self):
        script = """
import torch
import torch.nn as nn
from torch.profiler import supported_activities, profile
from torch.autograd.profiler import KinetoStepTracker

class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def payload(use_cuda=False):
    x = torch.randn(10, 10)
    if use_cuda:
        x = x.cuda()
    y = torch.randn(10, 10)
    if use_cuda:
        y = y.cuda()
    z = torch.mm(x, y)
    z = z + y
    if use_cuda:
        z = z.cpu()

niters = 8
use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
net = SimpleNet()
opt = torch.optim.SGD(net.parameters(), lr=0.01)
opt.zero_grad()
inputs = torch.rand(10)

with profile(activities=supported_activities()):
    payload(use_cuda=use_cuda)

initial_step = KinetoStepTracker.current_step()

def run_batch():
    out = net(inputs)
    loss = torch.nn.functional.cross_entropy(out, torch.rand(2))
    loss.backward()
    opt.step()

for _ in range(niters):
    run_batch()

with profile(
    activities=supported_activities(),
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
) as p:
    for _ in range(niters):
        run_batch()
        p.step()
if KinetoStepTracker.current_step() != initial_step + 2 * niters:
    raise AssertionError(f"Expected step {initial_step + 2 * niters}, got {KinetoStepTracker.current_step()}")
"""
        try:
            subprocess.check_output(
                [sys.executable, "-W", "always", "-c", script],
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            if e.returncode != 0:
                self.assertTrue(
                    False,
                    "Kineto is not working properly with the Dynolog environment variable",
                )

    def test_concrete_inputs_profiling(self):
        x = torch.rand(2, 6)
        with profile(record_shapes=True) as p:
            y = x.as_strided([4, 3], [1, 4])

        found = False
        for e in p.events():
            if e.name in ("aten::as_strided"):
                found = True
                self.assertTrue(len(e.input_shapes) > 0)
                self.assertTrue(len(e.concrete_inputs) > 0)
                self.assertEqual([2, 6], e.input_shapes[0])
                self.assertEqual([4, 3], e.concrete_inputs[1])
                self.assertEqual([1, 4], e.concrete_inputs[2])

        self.assertTrue(found, "Expected to find aten::as_strided but did not")

    def test_concrete_inputs_profiling_toggling(self):
        try:
            for before, after in [(True, False), (False, True)]:
                x = torch.rand(2, 6)
                torch._C._profiler._set_record_concrete_inputs_enabled_val(before)
                with profile(record_shapes=True) as p:
                    y = x.as_strided([4, 3], [1, 4])
                    torch._C._profiler._set_record_concrete_inputs_enabled_val(after)

                found = False
                for e in p.events():
                    if e.name in ("aten::as_strided"):
                        found = True
                        self.assertTrue(len(e.input_shapes))

                self.assertTrue(found, "Expected to find aten::as_strided but did not")
        finally:
            torch._C._profiler._set_record_concrete_inputs_enabled_val(True)

    def test_record_function_fast(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            for _ in range(4):
                # Test first with no optional args
                with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1"):
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf1"]), 4
        )
        for e in p.events():
            if e.name == "add_test_fast_rf1":
                self.assertTrue(e.input_shapes == [])
                self.assertTrue(e.kwinputs == {})
        with profile(record_shapes=True) as p:
            # add optional args
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf2", [x, y], {"stream": 0, "grid": "lambda x : x + 1"}
            )
            for _ in range(4):
                with cm:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf2"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf2":
                self.assertTrue(e.input_shapes == [[4, 4], [4, 4]])
                self.assertTrue(e.kwinputs == {"stream": 0, "grid": "lambda x : x + 1"})

        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf3", input_values=["hi"], keyword_values={"hi": "hello"}
            )
            for _ in range(4):
                try:
                    with cm:
                        x.add(y)
                        raise ValueError
                        x.relu()
                except ValueError:
                    pass

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf3"]), 4
        )
        self.assertFalse(any((e.name and "relu" in e.name) for e in p.events()))

        for e in p.events():
            if e.name == "add_test_fast_rf3":
                self.assertTrue(e.input_shapes == [[]])

        with profile() as p:
            for _ in range(4):
                with torch._C._profiler._RecordFunctionFast(
                    "add_test_fast_rf4", [x, y]
                ):
                    x.add(y)
                    with torch._C._profiler._RecordFunctionFast("add_test_fast_rf5"):
                        x.relu()

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf4"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf4":
                self.assertTrue(e.input_shapes == [])

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf5"]), 4
        )

        with profile(record_shapes=True) as p:
            # test optional args with tuple
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf6",
                (
                    x,
                    y,
                ),
            )
            for _ in range(4):
                with cm:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf6"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf6":
                self.assertTrue(e.input_shapes == [[4, 4], [4, 4]])

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_op_event_kwargs(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs",
                [x, y],
                {
                    "stream": 0,
                    "grid": "lambda x : x + 1",
                    "debug": 'debug"',
                    "boolean": True,
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("stream" in args)
                    self.assertTrue("grid" in args)
                    self.assertTrue("boolean" in args)
                    self.assertTrue(args["stream"] == 0)
                    self.assertTrue(args["grid"] == "lambda x : x + 1")
                    self.assertTrue(args["debug"] == "None")
                    self.assertTrue(args["boolean"])
                    self.assertTrue(e["cat"] == "cpu_op")

        with profile(record_shapes=True) as p1:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs",
                [x, y],
                {"stream": "test", "grid": [1, 2], "scope": "user_scope"},
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname1:
            p1.export_chrome_trace(fname1)
            with open(fname1) as f1:
                j = json.load(f1)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("stream" not in args)
                    self.assertTrue("grid" not in args)
                    self.assertTrue(e["cat"] == "user_annotation")

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_op_event_kwargs_list_of_strings(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs_string_list",
                [x, y],
                {
                    "string_list": ["hello", "world", "test"],
                    "int_param": 42,
                    "string_param": "single_string",
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs_string_list"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("string_list" in args)
                    self.assertTrue("int_param" in args)
                    self.assertTrue("string_param" in args)
                    # Check that the list of strings is properly serialized
                    # The list should be formatted as a JSON array by ivalueListToStr
                    self.assertEqual(args["string_list"], ["hello", "world", "test"])
                    self.assertEqual(args["int_param"], 42)
                    self.assertEqual(args["string_param"], "single_string")
                    self.assertTrue(e["cat"] == "cpu_op")

        # Test mixed types that should be filtered out
        with profile(record_shapes=True) as p1:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs_string_list_filtered",
                [x, y],
                {
                    "valid_string_list": ["valid1", "valid2"],
                    "mixed_list": ["string", 123],  # Should be filtered out
                    "non_string_list": [1, 2, 3],  # Should be filtered out
                    "valid_int": 100,
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname1:
            p1.export_chrome_trace(fname1)
            with open(fname1) as f1:
                j = json.load(f1)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs_string_list_filtered"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    # Only valid types should be present
                    self.assertTrue("valid_string_list" in args)
                    self.assertTrue("valid_int" in args)
                    # Invalid lists should be filtered out
                    self.assertTrue("mixed_list" not in args)
                    self.assertTrue("non_string_list" not in args)
                    # Check values
                    self.assertEqual(args["valid_string_list"], ["valid1", "valid2"])
                    self.assertEqual(args["valid_int"], 100)
                    self.assertTrue(e["cat"] == "cpu_op")

    def test_is_profiler_enabled(self):
        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        with profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        with torch.autograd.profiler.profile() as p:
            self.assertTrue(torch.autograd.profiler._is_profiler_enabled)

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

    def test_guarded_record_function_fast(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))

        with profile() as p:
            cm = torch._C._profiler._RecordFunctionFast("guarded_rf")
            for _ in range(4):
                if torch.autograd.profiler._is_profiler_enabled:
                    with cm:
                        x.add(y)
                else:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "guarded_rf"]), 4
        )

    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "requires gpu")
    def test_event_list(self):
        # AFAIK event list is part of legacy profiler and/or used when kineto is not available.
        # This test has basic sanity checks to test against obvious regressions.
        device = _current_accelerator_device_type()
        self.assertIsNotNone(device)
        x, y = (torch.rand((4, 4), requires_grad=True, device=device) for _ in range(2))
        with profile(with_stack=True) as p:
            z = (x @ y).relu().sum()
            z.backward()

        event_list = torch.autograd.profiler_util.EventList(p.events())
        # event_list._build_tree()

        with TemporaryFileName(mode="w+") as fname:
            event_list.export_chrome_trace(fname)
            with open(fname) as f:
                json.load(f)

        event_list.table()

    def _check_all_gpu_present(self, gpu_dict, max_gpu_count):
        for i in range(max_gpu_count):
            self.assertEqual(gpu_dict["GPU " + str(i)], 1)

    # Do json sanity testing. Checks that all events are between profiler start and end
    # also checks to see that GPU values are present in trace if cuda is used
    def _validate_basic_json(self, traceEvents, device_type="cpu"):
        MAX_GPU_COUNT = 8
        PROFILER_IDX = -7 if TEST_XPU else -4
        RECORD_END = -1
        RECORD_START = -5 if TEST_XPU else -2
        traceEventProfiler = traceEvents[PROFILER_IDX]

        self.assertTrue(traceEventProfiler["name"] == "PyTorch Profiler (0)")
        self.assertTrue(traceEvents[RECORD_END]["name"] == "Record Window End")
        self.assertTrue(
            traceEvents[RECORD_START]["name"] == "Iteration Start: PyTorch Profiler"
        )
        # check that the profiler starts/ends within the record interval
        self.assertGreaterEqual(
            traceEventProfiler["ts"],
            traceEvents[RECORD_START]["ts"],
            "Profiler starts before record!",
        )

        # Compare to nextafter value to avoid errors due to floating point precision
        RECORDS_END_TS = math.nextafter(traceEvents[RECORD_END]["ts"], math.inf)

        self.assertLessEqual(
            traceEventProfiler["ts"] + traceEventProfiler["dur"],
            RECORDS_END_TS,
            "Profiler ends after record end!",
        )

        gpu_dict = collections.defaultdict(int)
        for i, traceEvent in enumerate(traceEvents):
            if (
                i == len(traceEvents) + RECORD_END
                or i == len(traceEvents) + RECORD_START
            ):
                continue
            # make sure all valid trace events are within the bounds of the profiler
            if "ts" in traceEvent:
                self.assertGreaterEqual(
                    traceEvent["ts"],
                    traceEventProfiler["ts"],
                    "Trace event is out of bounds",
                )
            # some python events seem to go a little past record end probably because
            # of some clock inaccuracies so just compare events ending to RECORD_END
            tid = traceEvent.get("tid", "")
            if (
                "dur" in traceEvent
                and isinstance(tid, str)
                and "__xpu_profiler__" not in tid
            ):
                is_async_xpu_event = device_type == "xpu" and (
                    traceEvent.get("cat", "") in {"kernel", "gpu_memcpy"}
                    or "runtime" in traceEvent.get("cat", "")
                    or "GPU" in str(traceEvent.get("args", {}).get("labels", ""))
                )
                if not is_async_xpu_event:
                    self.assertLessEqual(
                        traceEvent["ts"] + traceEvent["dur"],
                        RECORDS_END_TS,
                        "Trace event ends too late!",
                    )
            gpu_value = traceEvent.get("args", {}).get("labels", None)
            if gpu_value and "GPU" in gpu_value:
                gpu_dict[gpu_value] += 1
                # Max PID offset is 5M, based from pytorch/kineto include header:
                # https://github.com/pytorch/kineto/blob/8681ff11e1fa54da39023076c5c43eddd87b7a8a/libkineto/include/output_base.h#L35
                kExceedMaxPid = 5000000
                self.assertTrue(
                    traceEvents[i + 1]["args"]["sort_index"]
                    == kExceedMaxPid + int(gpu_value.split()[1])
                )

        # TODO add checking gpu count if cpuOnly_ is true or not

    def _test_chrome_trace_basic_helper(self, device_type="cpu"):
        device = device_type
        x, y = (torch.rand(4, 4).to(device) for _ in range(2))

        with profile(with_stack=True) as p:
            torch.add(x, y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                report = json.load(f)
                self._validate_basic_json(report["traceEvents"], device_type)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_basic_chrome_trace(self):
        self._test_chrome_trace_basic_helper()
        if torch.accelerator.is_available():
            device_type = _current_accelerator_device_type()
            self.assertIsNotNone(device_type)
            self._test_chrome_trace_basic_helper(device_type=device_type)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_time_scale(self):
        MARGIN_ERROR = 0.5
        SEC_TO_US = 1000 * 1000
        WAIT_TIME = 10
        with profile() as p:
            with torch.profiler.record_function("test_span"):
                for _ in range(WAIT_TIME):
                    torch.rand(4, 4)
                    time.sleep(1)
        events = p.events()

        # make sure function events are scaled appropriately
        self.assertTrue(events[0].name == "test_span")
        test_span = events[0]
        self.assertGreaterEqual(
            test_span.cpu_time / SEC_TO_US,
            WAIT_TIME - MARGIN_ERROR,
            "event out of range",
        )
        self.assertLessEqual(
            test_span.cpu_time / SEC_TO_US,
            WAIT_TIME + MARGIN_ERROR,
            "event out of range",
        )

        # make sure tracing is scaled appropriately
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                report = json.load(f)
            events = report["traceEvents"]
            for event in events:
                if event["name"] == "test_span":
                    self.assertGreaterEqual(
                        event["dur"] / SEC_TO_US,
                        WAIT_TIME - MARGIN_ERROR,
                        "profiling out of range",
                    )
                    self.assertLessEqual(
                        event["dur"] / SEC_TO_US,
                        WAIT_TIME + MARGIN_ERROR,
                        "profiling out of range",
                    )

    def _schedule_helper(self, warmup, active, repeat, acc_events=True):
        with profile(
            schedule=torch.profiler.schedule(
                skip_first=0,
                wait=0,
                warmup=warmup,
                active=active,
                repeat=repeat,
            ),
            acc_events=acc_events,
        ) as prof:
            for _ in range(100):
                torch.add(1, 2)
                prof.step()
        # print(prof.key_averages())
        for ev in prof.key_averages():
            if ev.key == "aten::add":
                return ev.count
        return 0

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_schedule_function_count(self):
        self.assertEqual(self._schedule_helper(warmup=0, active=1, repeat=1), 1)
        self.assertEqual(self._schedule_helper(warmup=0, active=5, repeat=0), 100)
        self.assertEqual(self._schedule_helper(warmup=0, active=5, repeat=10), 50)
        self.assertEqual(self._schedule_helper(warmup=1, active=5, repeat=0), 83)
        self.assertEqual(self._schedule_helper(warmup=10, active=10, repeat=4), 40)
        self.assertEqual(self._schedule_helper(warmup=50, active=1, repeat=0), 1)
        self.assertEqual(
            self._schedule_helper(warmup=0, active=5, repeat=0, acc_events=False), 0
        )
        self.assertEqual(
            self._schedule_helper(warmup=10, active=10, repeat=4, acc_events=False), 10
        )

    def _step_helper_func(self, prof):
        time.sleep(0.1)
        torch.randn(1, 3, 224, 224)
        prof.step()

    def _partial_overlap(self, prof_step, step_helper_func):
        p_start = prof_step["ts"]
        p_end = prof_step["ts"] + prof_step["dur"]
        h_start = step_helper_func["ts"]
        h_end = step_helper_func["ts"] + step_helper_func["dur"]

        if p_start < h_start and p_end < h_end and p_end > h_start:
            return True
        if p_start > h_start and p_start < h_end and p_end > h_end:
            return True
        return False

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_cpu_annotation_overlap(self):
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
            experimental_config=torch._C._profiler._ExperimentalConfig(
                adjust_profiler_step=True
            ),
        ) as prof:
            for _ in range(5):
                self._step_helper_func(prof)
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            prof_steps = []
            step_helper_funcs = []
            with open(fname) as f:
                report = json.load(f)
                for event in report["traceEvents"]:
                    if "ProfilerStep" in event["name"]:
                        prof_steps.append(event)
                    if "step_helper_func" in event["name"]:
                        step_helper_funcs.append(event)
            self.assertEqual(len(prof_steps), 5)
            self.assertEqual(len(step_helper_funcs), 5)
            for i in range(len(step_helper_funcs)):
                for j in range(len(step_helper_funcs)):
                    self.assertTrue(
                        not self._partial_overlap(prof_steps[i], step_helper_funcs[j])
                    )

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_user_annotation(self):
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        with profile(activities=supported_activities()) as p:
            with torch.profiler.record_function("test_user_annotation"):
                self.payload(use_cuda=use_cuda)

        for evt in p.key_averages():
            if evt.key == "test_user_annotation":
                self.assertTrue(evt.is_user_annotation)
            else:
                self.assertFalse(evt.is_user_annotation)

    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "requires gpu")
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_basic_profile(self):
        # test a really basic profile to make sure no erroneous aten ops are run
        acc = torch.accelerator.current_accelerator()
        self.assertIsNotNone(acc)
        device = acc.type
        x = torch.randn(4, device=device)
        with torch.profiler.profile(with_stack=True) as p:
            x *= 2
        names = [e.name for e in p.events()]
        for name in names:
            if name.startswith("aten") and name != "aten::mul_":
                self.assertTrue(False, "Found unexpected event: " + name)
        self.assertTrue("aten::mul_" in names)

    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "requires gpu")
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_dynamic_toggle(self):
        acc = torch.accelerator.current_accelerator()
        self.assertIsNotNone(acc)
        device = acc.type
        gpu_activity = getattr(ProfilerActivity, device.upper(), None)
        self.assertIsNotNone(gpu_activity)
        activities = [ProfilerActivity.CPU, gpu_activity]
        with profile(activities=activities) as p:
            with torch.profiler.record_function("test_user_annotation"):
                x, y = (torch.rand(4, 4).to(device) for _ in range(2))
                torch.add(x, y)

        self.assertTrue(any("aten" in e.name for e in p.events()))

        self.assertTrue(any(device in e.name.lower() for e in p.events()))

        self.assertTrue(any("kernel" in e.name.lower() for e in p.events()))

        with profile(activities=activities) as p1:
            p1.toggle_collection_dynamic(False, [gpu_activity])
            with torch.profiler.record_function("test_user_annotation"):
                x, y = (torch.rand(4, 4).to(device) for _ in range(2))
                torch.add(x, y)

        self.assertTrue(any("aten" in e.name for e in p1.events()))

        self.assertTrue(all(device not in e.name for e in p1.events()))

        self.assertTrue(all("kernel" not in e.name.lower() for e in p1.events()))

        with profile(activities=activities) as p2:
            p2.toggle_collection_dynamic(False, activities)
            with torch.profiler.record_function("test_user_annotation"):
                x, y = (torch.rand(4, 4).to(device) for _ in range(2))
                torch.add(x, y)
        self.assertTrue(len(p2.events()) == 0)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_lazy_build_tree(self):
        with profile() as p:
            self.payload()

        stats = p._stats()
        # Test that the tree is not built
        self.assertEqual(stats.function_events_build_tree_call_duration_us, 0)
        self.assertEqual(stats.number_of_events, 0)

        # Test that the tree is built on demand
        p.events()
        self.assertGreater(stats.function_events_build_tree_call_duration_us, 0)
        self.assertGreater(stats.number_of_events, 0)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipIf(
        torch.get_device_module(GPU_TYPE).is_available(),
        "CUDA complains about forking after init",
    )
    @unittest.skipIf(torch.xpu.is_available(), "XPU complains about forking after init")
    @unittest.skipIf(IS_WINDOWS, "can't use os.fork() on Windows")
    def test_forked_process(self):
        # Induce a pid cache by running the profiler with payload
        def validate_forked_json(profiler):
            nonlocal cpu_op_found, parent_tid, child_pid
            with TemporaryFileName(mode="w+") as fname:
                profiler.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    for event in events:
                        if "cat" in event and event["cat"] == "cpu_op":
                            self.assertEqual(event["pid"], child_pid)
                            self.assertNotEqual(event["tid"], parent_tid)
                            cpu_op_found = True

        cpu_op_found = False
        parent_tid = threading.current_thread().ident
        with profile() as p:
            self.payload()
        pid = os.fork()
        if pid == 0:
            child_pid = os.getpid()
            with profile() as p:
                self.payload()
            validate_forked_json(p)
            self.assertTrue(cpu_op_found)
            os._exit(0)
        else:
            os.waitpid(pid, 0)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_skip_first_wait(self):
        # Other tests test when skip_first_wait is false (default) so just test the true case
        test_schedule = torch.profiler.schedule(
            skip_first=3, wait=5, warmup=1, active=2, repeat=2, skip_first_wait=1
        )
        test_schedule_expected_outputs = [
            # repeat No. 1 begin
            # skip first 3
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 1 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 1 end
            # repeat No. 1 end
            # ---
            # repeat No. 2 begin
            # wait 5
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            # warmup 1
            ProfilerAction.WARMUP,
            # active 2 begin
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            # active 2 end
            # repeat No. 2 end
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "requires gpu")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_disable_external_correlation(self):
        device = _current_accelerator_device_type()
        gpu_activity = _current_accelerator_activity()
        self.assertIsNotNone(device)
        self.assertIsNotNone(gpu_activity)
        activities = [ProfilerActivity.CPU, gpu_activity]

        def is_runtime_category(category):
            return category == "cuda_runtime" or "runtime" in category

        def check_correlations(event, disable_external_correlation):
            if "cat" in event and (
                event["cat"] in {"gpu_memcpy", "kernel"}
                or is_runtime_category(event["cat"])
            ):
                if disable_external_correlation:
                    self.assertTrue("External id" not in event["args"])
                else:
                    excluded_events = (
                        {"hipDeviceSynchronize"}
                        if TEST_WITH_ROCM
                        else {"cudaDeviceSynchronize"}
                    )
                    if event["name"] not in excluded_events:
                        self.assertTrue("External id" in event["args"])
                        self.assertTrue(event["args"]["External id"] > 0)

        def validate_json(prof, disable_external_correlation):
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    seen_event_types = set()
                    for event in events:
                        check_correlations(event, disable_external_correlation)
                        if "cat" in event:
                            seen_event_types.add(event["cat"])
                    self.assertTrue({"gpu_memcpy", "kernel"}.issubset(seen_event_types))
                    self.assertTrue(
                        any(is_runtime_category(cat) for cat in seen_event_types)
                    )

        # Run with External Id for device events on and off
        for disable_external_correlation in [False, True]:
            with profile(
                activities=activities,
                experimental_config=torch._C._profiler._ExperimentalConfig(
                    disable_external_correlation=disable_external_correlation
                ),
            ) as prof:
                self.payload(device_type=device, tensor_size=256)
            validate_json(prof, disable_external_correlation)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipUnless(TEST_CUDA or TEST_XPU, "requires gpu")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(
        "RelWithAssert" in torch.__config__.show(),
        "failing in debug build, see https://github.com/pytorch/pytorch/pull/150059 for example",
    )
    def test_profile_all_threads(self):
        profiling_started = threading.Event()
        profiling_ended = threading.Event()
        n_rep = 5
        device = _current_accelerator_device_type()
        self.assertIsNotNone(device)

        def prep_inputs():
            return [torch.randn(1024, 1024, device=device) for _ in range(2)]

        def main_thread_fn(profile_all_threads, returned_events):
            x, y = prep_inputs()
            experimental_config = torch._C._profiler._ExperimentalConfig(
                profile_all_threads=profile_all_threads
            )
            with torch.profiler.profile(
                experimental_config=experimental_config, record_shapes=True
            ) as p:
                profiling_started.set()
                for _ in range(n_rep):
                    _ = x @ y
                profiling_ended.wait()
            returned_events.append(p.events())

        def side_thread_fn():
            x, y = prep_inputs()
            profiling_started.wait()
            for _ in range(n_rep):
                _ = x @ y
            profiling_ended.set()

        def main_with_thread_fn(profile_all_threads):
            x, y = prep_inputs()
            experimental_config = torch._C._profiler._ExperimentalConfig(
                profile_all_threads=profile_all_threads
            )
            with torch.profiler.profile(
                experimental_config=experimental_config, record_shapes=True
            ) as p:
                side_thread = threading.Thread(target=side_thread_fn)
                side_thread.start()
                for _ in range(n_rep):
                    _ = x @ y
                side_thread.join()
            return p.events()

        for profile_all_threads in (True, False):
            returned_events = []
            main_thread = threading.Thread(
                target=main_thread_fn, args=(profile_all_threads, returned_events)
            )
            side_thread = threading.Thread(target=side_thread_fn)
            main_thread.start()
            side_thread.start()
            main_thread.join()
            side_thread.join()

            def verify_events(events):
                mm_events = collections.defaultdict(int)
                for e in events:
                    if e.name == "aten::mm":
                        mm_events[e.thread] += 1
                        self.assertEqual(e.input_shapes, [[1024, 1024], [1024, 1024]])
                self.assertEqual(len(mm_events), 1 + int(profile_all_threads))
                for v in mm_events.values():
                    self.assertEqual(v, n_rep)

            verify_events(returned_events[0])
            # test spawning thread from within the profiled region
            events = main_with_thread_fn(profile_all_threads)
            verify_events(events)

    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_python_gc_event(self):
        activities = [ProfilerActivity.CPU]

        def payload():
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            with record_function("pre_gc"):
                torch.mm(x, y)
            gc.collect()
            with record_function("post_gc"):
                torch.mm(x, y)

        def validate_json(prof, gc_collection_on):
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    # Find required events
                    if gc_collection_on:
                        pre_gc = next(
                            (e for e in events if e["name"] == "pre_gc"), None
                        )
                        post_gc = next(
                            (e for e in events if e["name"] == "post_gc"), None
                        )
                        python_gc_events = [
                            e for e in events if e["name"] == "Python GC"
                        ]
                        # Assert all required events are present
                        self.assertIsNotNone(pre_gc, "pre_gc event is missing")
                        self.assertIsNotNone(post_gc, "post_gc event is missing")
                        self.assertTrue(
                            len(python_gc_events) > 0, "No Python GC events found"
                        )
                        # Calculate boundaries
                        pre_gc_end = pre_gc["ts"] + pre_gc.get("dur", 0)
                        post_gc_start = post_gc["ts"]
                        # Assert at least one Python GC event is correctly placed.
                        # Other automatic GC events can happen while the profiler is
                        # active, especially during first-run initialization.
                        python_gc_events_between = [
                            e
                            for e in python_gc_events
                            if e["ts"] > pre_gc_end
                            and e["ts"] + e.get("dur", 0) < post_gc_start
                        ]
                        self.assertTrue(
                            len(python_gc_events_between) > 0,
                            "No Python GC events found between pre_gc and post_gc",
                        )
                    else:
                        python_gc_events = [
                            e for e in events if e["name"] == "Python GC"
                        ]
                        self.assertTrue(
                            len(python_gc_events) == 0,
                            "Python GC event found when flag of",
                        )

        for gc_flag in [True, False]:
            with profile(
                activities=activities,
                experimental_config=torch._C._profiler._ExperimentalConfig(
                    record_python_gc_info=gc_flag
                ),
                with_stack=True,
            ) as prof:
                payload()
            validate_json(prof, gc_flag)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_parse_kineto_results_timeout_none(self):
        """Test that _parse_kineto_results works normally without timeout."""
        with _profile(use_kineto=True) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        # Access function_events to trigger parsing
        events = p.function_events
        self.assertGreater(len(events), 0)
        self.assertTrue(any("aten::mm" in e.name for e in events))

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_parse_kineto_results_timeout_large(self):
        """Test that _parse_kineto_results with a large timeout processes all events."""
        with _profile(use_kineto=True) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        # Call _parse_kineto_results directly with a large timeout
        events_with_timeout = p._parse_kineto_results(p.kineto_results, timeout_s=60.0)
        events_without_timeout = p._parse_kineto_results(
            p.kineto_results, timeout_s=None
        )

        # Both should return the same number of events
        self.assertEqual(len(events_with_timeout), len(events_without_timeout))

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_parse_kineto_results_timeout_zero(self):
        """Test that _parse_kineto_results with zero timeout returns partial results and logs."""
        with _profile(use_kineto=True) as p:
            # Generate some events
            for _ in range(10):
                x = torch.randn(10, 10)
                y = torch.randn(10, 10)
                z = torch.mm(x, y)

        # Get baseline count without timeout
        events_no_timeout = p._parse_kineto_results(p.kineto_results, timeout_s=None)
        baseline_count = len(events_no_timeout)

        # With a zero timeout, we should get fewer events (or possibly zero)
        # and a warning should be logged
        import logging

        with self.assertLogs("torch.autograd.profiler", level=logging.WARNING) as cm:
            events_with_timeout = p._parse_kineto_results(
                p.kineto_results, timeout_s=0.0
            )

        # Check that we got a warning about timeout
        self.assertTrue(
            any("timed out" in msg and "partial results" in msg for msg in cm.output)
        )

        # With zero timeout, we should have fewer events than baseline
        self.assertLess(len(events_with_timeout), baseline_count)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_parse_kineto_results_timeout_fails(self):
        """Test that _parse_kineto_results fails with a negative timeout."""
        with _profile(use_kineto=True) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        with self.assertRaisesRegex(ValueError, "timeout_s must be non-negative"):
            events = p._parse_kineto_results(p.kineto_results, timeout_s=-60.0)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_public_api_post_processing_timeout_none(self):
        """Test that torch.profiler.profile works normally without timeout."""
        with profile() as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        events = p.events()
        self.assertGreater(len(events), 0)
        self.assertTrue(any("aten::mm" in e.name for e in events))

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_public_api_post_processing_timeout_large(self):
        """Test that torch.profiler.profile with a large timeout processes all events."""
        with profile(post_processing_timeout_s=60.0) as p:
            x = torch.randn(10, 10)
            y = torch.randn(10, 10)
            z = torch.mm(x, y)

        events = p.events()
        self.assertGreater(len(events), 0)
        self.assertTrue(any("aten::mm" in e.name for e in events))

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_public_api_post_processing_timeout_zero(self):
        """Test that torch.profiler.profile with zero timeout returns partial results."""
        import logging

        with profile(post_processing_timeout_s=0.0) as p:
            for _ in range(10):
                x = torch.randn(10, 10)
                y = torch.randn(10, 10)
                z = torch.mm(x, y)

        with self.assertLogs("torch.autograd.profiler", level=logging.WARNING) as cm:
            events = p.events()

        self.assertTrue(
            any("timed out" in msg and "partial results" in msg for msg in cm.output)
        )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_public_api_post_processing_timeout_fails(self):
        """Test that torch.profiler.profile with a negative timeout fails correctly."""
        with self.assertRaisesRegex(
            ValueError, "post_processing_timeout_s must be non-negative"
        ):
            with profile(post_processing_timeout_s=-1.0) as p:
                x = torch.randn(10, 10)
                y = torch.randn(10, 10)
                z = torch.mm(x, y)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_profiler(self):
        """Basic test for torch.profiler.profile API."""
        use_cuda = torch.profiler.ProfilerActivity.CUDA in supported_activities()
        activities = [ProfilerActivity.CPU]
        if use_cuda:
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities) as p:
            self.payload(use_cuda=use_cuda)
        events = p.events()
        self.assertGreater(len(events), 0)
        found_mm = False
        for e in events:
            if "aten::mm" in e.name:
                found_mm = True
        self.assertTrue(found_mm)
        if use_cuda:
            gpu_events = [e for e in events if e.device_type == DeviceType.CUDA]
            self.assertGreater(len(gpu_events), 0, "No GPU events captured by profiler")

    @unittest.skipIf(
        not torch.get_device_module(GPU_TYPE).is_available(), "requires CUDA"
    )
    @unittest.skipIf(TEST_WITH_ROCM, "not supported on ROCm")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_dict_syntax(self):
        """Dict syntax collects only the requested activity types."""
        with profile(
            activities=[{ProfilerActivity.CUDA: ["GPU_MEMCPY", "CUDA_RUNTIME"]}],
        ) as p:
            x = torch.randn(10, 10).to(GPU_TYPE)
            y = torch.mm(x, x)
        events = p.events()
        self.assertGreater(len(events), 0)
        print(events)
        # Verify we got GPU_MEMCPY events (HtoD copy from .to(GPU_TYPE)).
        has_memcpy = any("Memcpy" in e.name for e in events)
        self.assertTrue(has_memcpy, "Expected GPU_MEMCPY events")
        # Verify we got CUDA_RUNTIME events (e.g. cudaLaunchKernel).
        has_runtime = any(GPU_TYPE in e.name for e in events)
        self.assertTrue(has_runtime, "Expected CUDA_RUNTIME events")
        # OVERHEAD events (e.g. Lazy Function Loading) should NOT appear.
        has_overhead = any("Lazy Function Loading" in e.name for e in events)
        self.assertFalse(has_overhead)

    @unittest.skipIf(
        not torch.get_device_module(GPU_TYPE).is_available(), "requires CUDA"
    )
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_mixed_syntax(self):
        """Enum and dict entries can coexist for different activity groups."""
        activities = [ProfilerActivity.CPU, {ProfilerActivity.CUDA: ["GPU_MEMCPY"]}]
        with profile(activities=activities) as p:
            with record_function("test_annotation"):
                x = torch.randn(10, 10).to(GPU_TYPE)
                y = torch.mm(x, x)
        self.assertGreater(len(p.events()), 0)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_duplicate_raises(self):
        """Same activity appearing more than once raises ValueError."""
        with self.assertRaises(ValueError):
            with profile(
                activities=[
                    ProfilerActivity.CPU,
                    {ProfilerActivity.CPU: ["CPU_OP"]},
                ],
            ) as p:
                pass

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_invalid_type_name(self):
        """Invalid activity type name raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            with profile(
                activities=[{ProfilerActivity.CPU: ["NONEXISTENT_TYPE"]}],
            ) as p:
                x = torch.randn(10, 10)
                y = torch.mm(x, x)

    @unittest.skipIf(
        not torch.get_device_module(GPU_TYPE).is_available(), "requires CUDA"
    )
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_nonmember_type_name(self):
        """Activity type name that is not a member of the requested activity group raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            with profile(
                activities=[{ProfilerActivity.CUDA: ["CPU_OP"]}],
            ) as p:
                x = torch.randn(10, 10)
                y = torch.mm(x, x)

    @unittest.skipIf(
        not torch.get_device_module(GPU_TYPE).is_available(), "requires CUDA"
    )
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    def test_activity_filter_empty_list(self):
        """Passing an empty list to activities means not collecting for the specified activity."""
        with profile(
            activities=[{ProfilerActivity.CUDA: []}],
        ) as p:
            x = torch.randn(10, 10).to(GPU_TYPE)
            y = torch.mm(x, x)
        self.assertEqual(len(p.events()), 0)

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @unittest.skipIf(not TEST_CUDA, "CUDA is required")
    def test_kineto_kernel_metadata_in_trace(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self.payload(use_cuda=True)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)
            events = trace["traceEvents"]
            kernel_events = [e for e in events if e.get("cat", "") == "kernel"]
            self.assertGreater(
                len(kernel_events), 0, "Error: No kernel events in trace"
            )
            has_kernel_launch_metadata = False
            for ke in kernel_events:
                args = ke.get("args", {})
                name = ke.get("name", "<unknown>")
                for key in ["device", "stream", "correlation"]:
                    self.assertIn(key, args, f"kernel '{name}' missing '{key}'")
                # Some kernel events on ROCm (__amd_rocclr...) do not have grid/block metadata
                # so we just validate that it shows up for at least one event
                has_grid = "grid" in args
                has_block = "block" in args
                self.assertEqual(
                    has_grid,
                    has_block,
                    f"kernel '{name}' should provide grid and block together",
                )
                has_kernel_launch_metadata |= has_grid
            self.assertTrue(
                has_kernel_launch_metadata,
                "Error: No kernel events in trace contained grid/block metadata",
            )


class SimpleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class MockNode:
    def __init__(self, name, children) -> None:
        self.name = name
        self.children = [MockNode(name, i) for name, i in children.items()]


class TestExperimentalUtils(TestCase):
    def make_tree(self) -> list[MockNode]:
        tree = {
            "root_0": {
                "1": {"2": {}},
                "3": {
                    "4": {},
                    "5": {},
                },
            },
            "root_1": {
                "6": {},
                "7": {},
                "8": {
                    "9": {"10": {}},
                },
            },
        }
        return [MockNode(name, i) for name, i in tree.items()]

    def test_dfs(self) -> None:
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_dfs(self.make_tree())),
            "root_0 1 2 3 4 5 root_1 6 7 8 9 10",
        )

    def test_bfs(self) -> None:
        self.assertEqual(
            " ".join(i.name for i in _utils.traverse_bfs(self.make_tree())),
            "root_0 root_1 1 3 6 7 8 2 4 5 9 10",
        )

    @unittest.skipIf(
        not IS_LINUX or not (IS_X86 or IS_ARM64), "linux x86/aarch64 only cpp unwinding"
    )
    def test_fuzz_symbolize(self):
        # generate some random addresses in the text section and make sure the
        # symbolizers do not throw exceptions/crash
        def get_text_sections():
            text_sections = []
            seen = set()
            for filename in os.listdir("/proc/self/map_files"):
                library = os.readlink("/proc/self/map_files/" + filename)
                if ".so" not in library or library in seen:
                    continue
                seen.add(library)
                with open(os.path.join("/proc/self/map_files", library), "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

                    def unpack(fmt, offset):
                        return struct.unpack(
                            fmt, mm[offset : offset + struct.calcsize(fmt)]
                        )

                    if mm[:4] != b"\x7fELF":
                        continue
                    (section_headers_start,) = unpack("Q", 40)
                    (section_header_size,) = unpack("H", 58)
                    (num_section_headers,) = unpack("H", 60)
                    (shstrndx,) = unpack("H", 62)
                    (shstrtab_offset,) = unpack(
                        "Q", section_headers_start + shstrndx * section_header_size + 24
                    )
                    for i in range(num_section_headers):
                        (section_name_offset,) = unpack(
                            "I", section_headers_start + i * section_header_size
                        )
                        name_start = shstrtab_offset + section_name_offset
                        section_name = mm[name_start : name_start + 6]
                        if section_name != b".text\0":
                            continue
                        (section_offset,) = unpack(
                            "Q", section_headers_start + i * section_header_size + 24
                        )
                        (section_size,) = unpack(
                            "Q", section_headers_start + i * section_header_size + 32
                        )
                        start = int(filename.split("-")[0], 16) + section_offset
                        text_sections.append((start, section_size))
                        break
                    mm.close()
            return text_sections

        r = random.Random()
        r.seed(1)
        text_sections = get_text_sections()
        addrs = []
        for _ in range(200):
            s = r.randrange(0, len(text_sections))
            start, size = text_sections[s]
            addr = r.randrange(start, start + size)
            addrs.append(addr)
        fast = torch._C._profiler.symbolize_addresses(addrs, "fast")
        dladdr = torch._C._profiler.symbolize_addresses(addrs, "dladdr")
        addr2line = torch._C._profiler.symbolize_addresses(addrs, "addr2line")
        self.assertEqual(len(fast), len(addrs))
        self.assertEqual(len(addr2line), len(fast))

    def test_profiler_overload_names(self):
        from torch.library import _scoped_library, fallthrough_kernel

        def validate_json(prof):
            print()
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    self.assertTrue(
                        any("aten::add.Tensor" in e["name"] for e in events)
                    )
                    self.assertTrue(any("aten::add.out" in e["name"] for e in events))

        with _scoped_library("aten", "IMPL") as my_lib:
            my_lib.impl("add.Tensor", fallthrough_kernel, "CPU")
            experimental_config = torch._C._profiler._ExperimentalConfig(
                capture_overload_names=True
            )
            with profile(
                experimental_config=experimental_config,
                activities=[ProfilerActivity.CPU],
            ) as prof:
                torch.add(1, 5)

            # The following execution trace is expected
            #
            # Dispatch trace:
            # [call] op=[aten::add.Tensor], key=[AutogradCPU]
            #   [redispatch] op=[aten::add.Tensor], key=[Undefined]
            #     [call] op=[aten::empty.memory_format], key=[BackendSelect]
            #       [redispatch] op=[aten::empty.memory_format], key=[CPU]
            #     [call] op=[aten::add.out], key=[CPU]
            #
            # prof.table()
            # ---------------  ---------------  ------------  ------------  ------------  ------------  ------------  ------------
            #            Name    Overload Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
            # ---------------  ---------------  ------------  ------------  ------------  ------------  ------------  ------------
            #       aten::add           Tensor        71.97%     130.887us       100.00%     181.873us     181.873us             1
            #     aten::empty    memory_format         8.52%      15.489us         8.52%      15.489us      15.489us             1
            #       aten::add              out        19.52%      35.497us        19.52%      35.497us      35.497us             1
            # ---------------  ---------------  ------------  ------------  ------------  ------------  ------------  ------------

            # aten::add.out and aten::empty.memory_format are children of aten::add.Tensor
            aten_add_parent: list[FunctionEvent] = [
                event for event in prof.events() if len(event.cpu_children) == 2
            ]
            if len(aten_add_parent) != 1:
                raise AssertionError(
                    f"Expected 1 parent event, got {len(aten_add_parent)}"
                )
            aten_add_parent = aten_add_parent[0]
            if aten_add_parent.overload_name != "Tensor":
                raise AssertionError(
                    f"Expected overload_name 'Tensor', got '{aten_add_parent.overload_name}'"
                )

            aten_add_out_event = [
                c for c in aten_add_parent.cpu_children if c.overload_name == "out"
            ]
            if len(aten_add_out_event) != 1:
                raise AssertionError(
                    f"Expected 1 out event, got {len(aten_add_out_event)}"
                )

            # Without group_by_overload_name, the overload name is ignored in the key averages
            key_averages = prof.key_averages()
            if len(key_averages) != 2:
                raise AssertionError(
                    f"Expected 2 key averages, got {len(key_averages)}"
                )
            if "Overload Name" in key_averages.table():
                raise AssertionError("Overload Name should not be in table")

            key_averages = prof.key_averages(group_by_overload_name=True)
            if len(key_averages) != 3:
                raise AssertionError(
                    f"Expected 3 key averages with group_by_overload_name, got {len(key_averages)}"
                )
            if "Overload Name" not in key_averages.table():
                raise AssertionError("Overload Name should be in table")
            validate_json(prof)

    def test_expose_kineto_event_metadata(self):
        def check_metadata(prof, op_name, metadata_key):
            with TemporaryFileName(mode="w+") as fname:
                prof.export_chrome_trace(fname)
                with open(fname) as f:
                    events = json.load(f)["traceEvents"]
                    found_op = False
                    for e in events:
                        if "name" in e and "args" in e and e["name"] == op_name:
                            if metadata_key not in e["args"]:
                                raise AssertionError(
                                    f"Metadata for '{op_name}' in Chrome trace did not contain '{metadata_key}'."
                                )
                            found_op = True
                    if not found_op:
                        raise AssertionError(
                            f"Could not find op '{op_name}' in Chrome trace."
                        )
                found_op = False
                for event in prof.events():
                    if event.name == op_name:
                        if metadata_key not in event.metadata_json:
                            raise AssertionError(
                                f"Metadata for '{op_name}' in FunctionEvent did not contain '{metadata_key}'."
                            )
                        found_op = True
                if not found_op:
                    raise AssertionError(
                        f"Could not find op '{op_name}' in prof.events()."
                    )

        experimental_config = torch._C._profiler._ExperimentalConfig(
            expose_kineto_event_metadata=True
        )
        with profile(
            experimental_config=experimental_config,
            activities=[ProfilerActivity.CPU],
        ) as prof:
            torch.add(1, 5)

        check_metadata(prof, op_name="aten::add", metadata_key="Ev Idx")

    @unittest.skipIf(
        not torch.get_device_module(GPU_TYPE).is_available(), "requires CUDA"
    )
    def test_profiler_debug_autotuner(self):
        """
        This test makes sure that profiling events will be present when the kernel is run using the DebugAutotuner.
        """
        if not is_big_gpu():
            raise unittest.SkipTest("requires large gpu to max-autotune")
        in1 = torch.randn((256, 512), device=GPU_TYPE, dtype=torch.float16)
        in2 = torch.randn((512, 768), device=GPU_TYPE, dtype=torch.float16)

        def mm():
            return torch.mm(in1, in2)

        pb_mm = torch.compile(
            mm,
            options={
                "benchmark_kernel": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "profile_bandwidth": True,
            },
        )
        comp_mm = torch.compile(
            mm,
            options={
                "benchmark_kernel": True,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )

        with profile() as prof1:
            pb_mm()
        with profile() as prof2:
            comp_mm()

        def names(prof):
            return {
                ev.name
                for ev in prof.events()
                if "mm" in ev.name or "triton" in ev.name
            }

        n1 = names(prof1)
        n2 = names(prof2)
        self.assertEqual(n1, n2)


class TestPrivateUse1ProfilerState(TestCase):
    """Tests for PrivateUse1 profiler state selection logic."""

    def test_kineto_privateuse1_state_with_use_kineto_true(self):
        """Test that KINETO_PRIVATEUSE1 state is selected when use_kineto=True."""
        from unittest.mock import patch

        from torch._C._profiler import ProfilerState

        with patch(
            "torch.autograd.profiler._get_privateuse1_backend_name",
            return_value="custom_backend",
        ):
            prof = _profile(
                use_cpu=True,
                use_device="custom_backend",
                use_kineto=True,
            )
            self.assertEqual(prof.profiler_kind, ProfilerState.KINETO_PRIVATEUSE1)

    def test_kineto_privateuse1_fallback_state_with_use_kineto_false(self):
        """Test that KINETO_PRIVATEUSE1_FALLBACK is selected when use_kineto=False."""
        from unittest.mock import patch

        from torch._C._profiler import ProfilerState

        with patch(
            "torch.autograd.profiler._get_privateuse1_backend_name",
            return_value="custom_backend",
        ):
            prof = _profile(
                use_cpu=True,
                use_device="custom_backend",
                use_kineto=False,
            )
            self.assertEqual(
                prof.profiler_kind, ProfilerState.KINETO_PRIVATEUSE1_FALLBACK
            )

    def test_privateuse1_fallback_requires_use_cpu(self):
        """Test that KINETO_PRIVATEUSE1_FALLBACK requires use_cpu=True."""
        from unittest.mock import patch

        with patch(
            "torch.autograd.profiler._get_privateuse1_backend_name",
            return_value="custom_backend",
        ):
            # When use_kineto=False and use_cpu=False, should raise AssertionError
            with self.assertRaises(AssertionError):
                _profile(
                    use_cpu=False,
                    use_device="custom_backend",
                    use_kineto=False,
                )


@unittest.skipIf(not kineto_available(), "Kineto is required")
@unittest.skipIf(
    not torch.get_device_module(GPU_TYPE).is_available(), "CUDA is required"
)
class TestProfilerEventsParity(TestCase):
    """Tests validating parity between events() and export_chrome_trace() JSON."""

    def test_python_function_events_in_events(self):
        class DummyModule(nn.Module):
            def forward(self, x):
                return x + 1

        mod = DummyModule()
        with profile(
            activities=[ProfilerActivity.CPU],
            with_stack=True,
            experimental_config=_ExperimentalConfig(verbose=True),
        ) as prof:
            mod(torch.randn(4, 4))

        events = prof.events()
        python_events = [e for e in events if e.is_python_function]
        self.assertGreater(len(python_events), 0)
        for e in python_events:
            self.assertIsInstance(e.name, str)
            self.assertGreater(e.time_range.end - e.time_range.start, 0)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

            json_py = [
                e
                for e in trace["traceEvents"]
                if e.get("cat") == "python_function" and e.get("ph") == "X"
            ]
            self.assertEqual(len(python_events), len(json_py))

            # Verify python_id/parent_id/module_id parity with JSON args
            fe_mod = next((e for e in events if "DummyModule" in e.name), None)
            self.assertIsNotNone(fe_mod)
            self.assertGreater(fe_mod.python_id, 0)
            self.assertGreaterEqual(fe_mod.python_module_id, 0)

            json_mod = next(
                (e for e in json_py if "DummyModule" in e.get("name", "")),
                None,
            )
            self.assertIsNotNone(json_mod)
            args = json_mod["args"]
            self.assertEqual(fe_mod.python_id, args["Python id"])
            self.assertEqual(fe_mod.python_parent_id, args["Python parent id"])
            self.assertEqual(fe_mod.python_module_id, args["Python module id"])

    def test_profiler_flow_events_parity(self):
        """Verify that async CPU->GPU flow fields on events() match Chrome trace JSON."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            x = torch.randn(32, 32, device=GPU_TYPE)
            torch.mm(x, x)

        # Collect async CPU->GPU flow info from events()
        events_with_flow = [
            e for e in prof.events() if e.flow_id is not None and e.flow_id != 0
        ]
        self.assertGreater(
            len(events_with_flow), 0, "No flow events found via events()"
        )

        for e in events_with_flow:
            self.assertIsInstance(e.flow_id, int)
            self.assertIsInstance(e.flow_type, int)
            self.assertIsInstance(e.flow_start, bool)

        # Verify parity with Chrome trace JSON for async CPU->GPU flow
        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_flow_events = [
                e
                for e in j["traceEvents"]
                if e.get("ph") in ("s", "f") and e.get("cat") == "ac2g"
            ]
            json_flow_starts = {e["id"] for e in json_flow_events if e["ph"] == "s"}
            json_flow_ends = {e["id"] for e in json_flow_events if e["ph"] == "f"}

            # kLinkAsyncCpuGpu = 2
            ac2g_events = [e for e in events_with_flow if e.flow_type == 2]
            events_flow_starts = {e.flow_id for e in ac2g_events if e.flow_start}
            events_flow_ends = {e.flow_id for e in ac2g_events if not e.flow_start}

            self.assertEqual(
                json_flow_starts,
                events_flow_starts,
                "Async CPU->GPU flow start IDs differ between events() and Chrome trace",
            )
            self.assertEqual(
                json_flow_ends,
                events_flow_ends,
                "Async CPU->GPU flow end IDs differ between events() and Chrome trace",
            )

    def test_profiler_fwdbwd_flow_events_parity(self):
        """Verify that fwd->bwd flow fields on events() match Chrome trace JSON."""
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            t1 = torch.ones(1, requires_grad=True)
            t2 = torch.ones(1, requires_grad=True)
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            loss.backward()

        fwdbwd_events = [
            e for e in prof.events() if e.flow_type == 1 and e.flow_id != 0
        ]
        self.assertGreater(
            len(fwdbwd_events), 0, "No fwdbwd flow events found via events()"
        )

        events_flow_starts = {e.flow_id for e in fwdbwd_events if e.flow_start}
        events_flow_ends = {e.flow_id for e in fwdbwd_events if not e.flow_start}

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_flow_events = [
                e
                for e in j["traceEvents"]
                if e.get("ph") in ("s", "f") and e.get("cat") == "fwdbwd"
            ]
            json_flow_starts = {e["id"] for e in json_flow_events if e["ph"] == "s"}
            json_flow_ends = {e["id"] for e in json_flow_events if e["ph"] == "f"}

            self.assertEqual(
                json_flow_starts,
                events_flow_starts,
                "fwdbwd flow start IDs differ between events() and Chrome trace",
            )
            self.assertEqual(
                json_flow_ends,
                events_flow_ends,
                "fwdbwd flow end IDs differ between events() and Chrome trace",
            )

    def test_profiler_timestamp_consistency(self):
        """Verify that FunctionEvent timestamps can reconstruct Chrome trace ts values."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            x = torch.randn(32, 32, device=GPU_TYPE)
            torch.mm(x, x)

        trace_start_ns = prof.profiler.kineto_results.trace_start_ns()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            # Chrome trace is relative to a different base time which is not exposed in Python.
            # It's probably not important to do so as we still have the relative differences
            # in duration.
            base_time_ns = j.get("baseTimeNanoseconds", 0)

            # Grab mm timestamp from events() and json
            fe_mm = next((e for e in prof.events() if e.name == "aten::mm"), None)
            json_mm = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::mm" and e.get("ph") == "X"
                ),
                None,
            )

            # Reconstruct Chrome trace ts from events():
            # absolute_ns = mm_op_start_us * 1000 + trace_start_ns
            # chrome_ts = (absolute_ns - base_time_ns) / 1000 -> realign with json timeframe
            absolute_ns = int(fe_mm.time_range.start * 1000) + trace_start_ns
            recovered_ts = (absolute_ns - base_time_ns) / 1000
            self.assertEqual(
                recovered_ts,
                json_mm["ts"],
                msg="Recovered Chrome trace ts doesn't match JSON for aten::mm",
            )

    def test_profiler_op_args_events_parity(self):
        """Verify that cpu_op args on events() match Chrome trace JSON args."""
        base_tensor = torch.randn(1024, dtype=torch.float32)
        a = base_tensor.as_strided((16, 16), (17, 1), 0)
        b = base_tensor.as_strided((16, 16), (25, 2), 272)
        t1 = torch.ones((64, 32))
        t2 = torch.ones((64, 32))
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            torch.add(a, b)
            torch.cat([t1, t2])

        fe_add = next((e for e in prof.events() if e.name == "aten::add"), None)
        self.assertIsNotNone(fe_add)
        fe_cat = next((e for e in prof.events() if e.name == "aten::cat"), None)
        self.assertIsNotNone(fe_cat)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_add = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::add" and e.get("cat") == "cpu_op"
                ),
                None,
            )
            self.assertIsNotNone(json_add)
            args = json_add["args"]
            self.assertEqual(fe_add.structured_input_shapes, args["Input Dims"])
            self.assertEqual(fe_add.structured_input_strides, args["Input Strides"])
            self.assertEqual(fe_add.input_dtypes, args["Input type"])

            # Test a case with TensorList inputs -- structured_input_shapes
            # should handle TensorList nesting correctly.
            json_cat = next(
                (
                    e
                    for e in j["traceEvents"]
                    if e.get("name") == "aten::cat" and e.get("cat") == "cpu_op"
                ),
                None,
            )
            self.assertIsNotNone(json_cat)
            args_cat = json_cat["args"]
            self.assertEqual(fe_cat.structured_input_shapes, args_cat["Input Dims"])
            self.assertEqual(fe_cat.structured_input_strides, args_cat["Input Strides"])
            self.assertEqual(fe_cat.input_dtypes, args_cat["Input type"])

    def test_profiler_external_id_parity(self):
        """Verify that FunctionEvent.external_id matches External id in Chrome trace JSON."""
        from collections import Counter

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("test_region"):
                x = torch.randn(32, 32, device=GPU_TYPE)
                y = torch.mm(x, x)
                z = y + x
                z.cpu()
                torch.get_device_module(GPU_TYPE).synchronize()

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_name_ext = Counter(
                (e["name"], e["args"]["External id"])
                for e in j["traceEvents"]
                if e.get("args", {}).get("External id") is not None
            )
            events_name_ext = Counter(
                (ev.name, ev.external_id) for ev in prof.events() if ev.external_id != 0
            )

            self.assertEqual(
                events_name_ext,
                json_name_ext,
                "(name, external_id) pairs differ between events() and Chrome trace JSON",
            )

    def test_profiler_activity_type_parity(self):
        """Verify activity_type on events() matches Chrome trace cat field."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            x = torch.randn(32, 32, device=GPU_TYPE)
            torch.mm(x, x)

        events = prof.events()
        for e in events:
            self.assertIsInstance(e.activity_type, str)
            self.assertGreater(len(e.activity_type), 0)

        mm_event = next((e for e in events if e.name == "aten::mm"), None)
        self.assertIsNotNone(mm_event)
        self.assertEqual(mm_event.activity_type, "cpu_op")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)

            json_name_cats = {
                (e["name"], e["cat"])
                for e in j["traceEvents"]
                if e.get("ph") == "X" and "cat" in e
            }
            for e in events:
                self.assertIn(
                    (e.name, e.activity_type),
                    json_name_cats,
                    f"activity_type mismatch for {e.name}",
                )

    def test_structured_metadata_matches_chrome_trace(self):
        # Compare metadata fields between events() and Chrome trace JSON to make sure they stay in parity
        # 1. Run a dummy workload with profiling enabled and collect the json/events() outputs
        # 2. Parse each event instance in the json and events() to create a key->value mapping
        #      - The key is a tuple of metadata fields that should be unique for each event
        #      - The value is a dict of metadata fields for that event
        # 3. Ensure that the keys and values match between the json and events() outputs

        from torch.autograd.profiler_util import _EVENT_METADATA_KEYS

        target_cats = ("cuda_runtime", "gpu_memcpy", "kernel")
        allowed_non_structured_trace_keys = {
            "External id",
            "correlation",
            "cbid",
            "cid",
            "device",
            "kind",
            "kernel",
            "ptr",
            "src",
            "dst",
        }
        supported_trace_keys = set(_EVENT_METADATA_KEYS).union(
            allowed_non_structured_trace_keys
        )

        def metadata_dict_from_trace_args(args):
            out = {}
            for kineto_key, (field_name, convert) in _EVENT_METADATA_KEYS.items():
                if kineto_key in args:
                    raw_value = args[kineto_key]
                    out[field_name] = (
                        convert(raw_value) if isinstance(raw_value, str) else raw_value
                    )
            return out

        def metadata_dict_from_function_event(fe):
            if fe.event_metadata is None:
                return {}

            out = {}
            for field_name, _ in _EVENT_METADATA_KEYS.values():
                val = getattr(fe.event_metadata, field_name)
                if val is not None:
                    out[field_name] = val
            return out

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(
                expose_kineto_event_metadata=True
            ),
        ) as prof:
            x = torch.randn(10, 10, device=GPU_TYPE)
            y = torch.mm(x, x)
            z = x + y
            z.cpu()

        # Build a mapping from key to events() FunctionEvent metadata
        event_records = {}
        for fe in prof.events():
            if fe.external_id == 0 or fe.id == 0 or fe.activity_type not in target_cats:
                continue
            # Using just one of these keys could result in collisions, so try to uniquely identify the event with all of them
            key = (fe.name, fe.activity_type, fe.external_id, fe.id)
            self.assertNotIn(
                key,
                event_records,
                f"Duplicate FunctionEvent record key encountered: {key}",
            )
            event_records[key] = metadata_dict_from_function_event(fe)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                trace = json.load(f)

        json_records = {}
        # Track unexpected (event_name, cat, key) combos, deduplicated
        unexpected_combos: set[tuple[str, str, str]] = set()

        # Loop through the trace events to perform a comparison
        for te in trace["traceEvents"]:
            cat = te.get("cat", "")
            args = te.get("args", {})
            ext_id = args.get("External id")
            correlation = args.get("correlation")

            if ext_id is None or correlation is None:
                continue
            if cat not in target_cats:
                continue

            # Any metadata keys that show up in JSON should show up in events()
            for k in set(args) - supported_trace_keys:
                unexpected_combos.add((te["name"][:100], cat, k))

            # Build the same key from JSON to try to match with a FunctionEvent
            key = (te["name"], te["cat"], ext_id, correlation)
            self.assertNotIn(
                key,
                json_records,
                f"Duplicate Chrome trace record key encountered: {key}",
            )
            json_records[key] = metadata_dict_from_trace_args(args)

        failure_msg = """\
====================================================================================
IMPORTANT: Are you making a Kineto change or bumping the third_party/kineto
submodule hash and seeing this message?

New metadata keys (see below) were found in the Chrome trace JSON that are not
yet exposed through the profiler's events() API (i.e. EventMetadata in
torch/autograd/profiler_util.py).

To fix this properly, you need to make sure the new Kineto data makes its way
to the events() property. The steps are:

1. Add the new key(s) to _EVENT_METADATA_KEYS in torch/autograd/profiler_util.py
   with the appropriate field name and type converter.
2. Add corresponding field(s) to the EventMetadata dataclass in the same file.
3. If the key should NOT be mapped (e.g. it duplicates an existing FunctionEvent
   attribute), add it to allowed_non_structured_trace_keys in this test instead.

For a model PR to follow, see: https://github.com/pytorch/pytorch/pull/180100
===================================================================================="""
        if unexpected_combos:
            summary = "\n".join(
                f"  {name} ({cat}): {key!r}"
                for name, cat, key in sorted(unexpected_combos)
            )
            raise AssertionError(f"\n{failure_msg}\n\nUnmapped keys:\n{summary}")

        self.assertGreater(len(json_records), 0, "No device-side records were compared")
        self.assertEqual(
            set(event_records),
            set(json_records),
            "Device event identities differ between events() and Chrome trace JSON",
        )

        for key in json_records:
            expected_meta = json_records[key]
            actual_meta = event_records[key]
            self.assertEqual(
                actual_meta,
                expected_meta,
                f"{key}: structured metadata differs between events() and Chrome trace JSON",
            )


if __name__ == "__main__":
    run_tests()
