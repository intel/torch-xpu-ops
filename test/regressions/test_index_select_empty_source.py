# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import subprocess
import sys

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestIndexSelectEmptySource(TestCase):
    @dtypes(torch.int32, torch.int64)
    @parametrize("use_out", [False, True])
    def test_zero_axis_and_empty_index(self, device, dtype, use_out):
        source = torch.empty((0, 3), device=device)
        index = torch.tensor([0, 1], dtype=dtype, device=device)
        kwargs = {"out": torch.empty(0, device=device)} if use_out else {}
        with self.assertRaisesRegex(
            RuntimeError, "self indexing axis dim should be positive"
        ):
            torch.index_select(source, 0, index, **kwargs)
        if use_out:
            kwargs["out"].resize_(0)
        index = index[:0]
        result = torch.index_select(source, 0, index, **kwargs)
        self.assertEqual(result.shape, source.shape)

    @dtypes(torch.int32, torch.int64)
    @parametrize("use_out", [False, True])
    @parametrize("shape,dim", [((0, 3), -1), ((3, 0), 0), ((2, 0, 3), 2)])
    @parametrize("layout", ["contiguous", "strided", "expanded", "scalar"])
    def test_valid_bounds(self, device, dtype, use_out, shape, dim, layout):
        source = torch.empty(shape, device=device)
        index = torch.tensor([0, 2], dtype=dtype, device=device)
        if layout == "strided":
            index = torch.tensor([0, 99, 2, 99], dtype=dtype, device=device)[::2]
        elif layout == "expanded":
            index = index[1:].expand(1025)
        elif layout == "scalar":
            index = index[1]
        kwargs = {"out": torch.empty(0, device=device)} if use_out else {}
        expected = torch.index_select(source.cpu(), dim, index.cpu())
        actual = torch.index_select(source, dim, index, **kwargs)
        self.assertEqual(actual.cpu(), expected)
        if use_out:
            self.assertIs(actual, kwargs["out"])

    @dtypes(torch.int32, torch.int64)
    @parametrize("use_out", [False, True])
    def test_graph_capture_and_replay(self, device, dtype, use_out):
        source = torch.empty((0, 3), device=device)
        index = torch.zeros(4, dtype=dtype, device=device)[::2]
        out = torch.empty((0, 2), device=device)
        anchor = torch.ones(1, device=device)
        stream = torch.xpu.Stream()
        stream.wait_stream(torch.xpu.current_stream())

        def select():
            if use_out:
                return torch.index_select(source, 1, index, out=out)
            return torch.index_select(source, 1, index)

        with torch.xpu.stream(stream):
            for _ in range(3):
                select()
        stream.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            result = select()
            captured_anchor = anchor + 1
        for value in (0, 2):
            index.fill_(value)
            torch.xpu.synchronize()
            graph.replay()
            torch.xpu.synchronize()
            self.assertEqual(result.shape, (0, 2))
            self.assertEqual(captured_anchor.cpu(), torch.full((1,), 2.0))

    @dtypes(torch.int32, torch.int64)
    def test_no_host_scalar_reads(self, device, dtype):
        source = torch.empty((0, 3), device=device)
        index = torch.tensor([0, 2], dtype=dtype, device=device)
        torch.index_select(source, 1, index)
        torch.xpu.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            torch.index_select(source, 1, index)
            torch.xpu.synchronize()
        names = {event.key for event in profile.key_averages()}
        self.assertNotIn("aten::item", names)
        self.assertNotIn("aten::_local_scalar_dense", names)
        self.assertNotIn("aten::aminmax", names)

    @dtypes(torch.int32, torch.int64)
    @parametrize("bad_index", [-1, 3, 2**32 + 1])
    @parametrize("use_out", [False, True])
    @parametrize("graph_replay", [False, True])
    def test_invalid_bounds(self, device, dtype, bad_index, use_out, graph_replay):
        if dtype == torch.int32 and bad_index > torch.iinfo(dtype).max:
            self.skipTest("Index value requires int64")
        # Device assertions abort the process, so isolate every invalid case.
        script = f"""
import torch
source = torch.empty((0, 3), device={device!r})
backing = torch.zeros(2050, dtype={dtype}, device={device!r})
index = backing[::2]
out = torch.empty((0, index.numel()), device={device!r})
def select():
    return torch.index_select(source, 1, index, **({{'out': out}} if {use_out} else {{}}))
if {graph_replay}:
    stream = torch.xpu.Stream()
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        select()
    stream.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        select()
index[-1] = {bad_index}
torch.xpu.synchronize()
print("checking invalid index", flush=True)
if {graph_replay}:
    graph.replay()
else:
    select()
torch.xpu.synchronize()
"""
        process = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = process.stdout + process.stderr
        self.assertNotEqual(process.returncode, 0, output)
        self.assertIn("checking invalid index", output)
        self.assertIn("index_value >= 0 && index_value < axis_size_", output)


instantiate_device_type_tests(
    TestIndexSelectEmptySource, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
