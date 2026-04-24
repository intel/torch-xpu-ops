# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import json
import tempfile
import unittest
from collections import defaultdict

import torch
from torch.profiler import DeviceType
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._triton import has_triton


class TestProfilerCorrectness(TestCase):
    ACTIVITIES = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.XPU,
    ]

    def _gen_and_check_json(self, prof, expected_cats=None):
        """Export chrome trace and validate JSON structure.

        Checks that the trace contains well-formed events with required fields.
        Optionally checks for the presence of specific event categories.
        Returns (count_names, count_cats) dicts for further assertions.
        """
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as tmp:
            prof.export_chrome_trace(tmp.name)

            with open(tmp.name) as f:
                data = json.load(f)

            self.assertIn("traceEvents", data)

            trace_events = data["traceEvents"]
            self.assertIsInstance(trace_events, list)
            self.assertGreater(len(trace_events), 0)

            count_names = defaultdict(int)
            count_cats = defaultdict(int)
            for event in trace_events:
                self.assertIn("ph", event)
                self.assertIn("name", event)

                if event["ph"] == "X":
                    self.assertIn("cat", event)
                    self.assertIn("dur", event)
                    self.assertIn("ts", event)
                    self.assertGreaterEqual(
                        event["dur"],
                        0,
                        f"Event '{event['name']}' has negative duration: {event['dur']}",
                    )
                    self.assertGreaterEqual(
                        event["ts"],
                        0,
                        f"Event '{event['name']}' has negative timestamp: {event['ts']}",
                    )
                    count_names[event["name"]] += 1
                    count_cats[event["cat"]] += 1

            if expected_cats:
                for cat in expected_cats:
                    self.assertIn(
                        cat,
                        count_cats,
                        f"Expected category '{cat}' not found in trace. "
                        f"Found categories: {list(count_cats.keys())}",
                    )

        return count_names, count_cats

    def _assert_has_device_types(self, prof, expected_types=None):
        """Assert that profiler events contain the expected device types."""
        if expected_types is None:
            expected_types = {DeviceType.CPU, DeviceType.XPU}

        device_types = defaultdict(int)
        for event in prof.events():
            device_types[event.device_type] += 1

        for dt in expected_types:
            self.assertIn(
                dt,
                device_types,
                f"Expected device type {dt} not found in events. "
                f"Found: {list(device_types.keys())}",
            )

    def _assert_key_averages_nonempty(self, prof):
        """Assert that key_averages() produces a non-empty table."""
        averages = prof.key_averages()
        self.assertGreater(len(averages), 0, "key_averages() returned no events")
        table = averages.table(sort_by="self_device_time_total", row_limit=-1)
        self.assertGreater(len(table), 0, "key_averages().table() is empty")
        return averages

    def _load_chrome_trace(self, prof):
        """Export chrome trace and return parsed JSON data."""
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=True) as tmp:
            prof.export_chrome_trace(tmp.name)
            with open(tmp.name) as f:
                return json.load(f)

    def _assert_has_positive_xpu_time(self, averages, msg=None):
        """Assert that at least one event has positive self_device_time_total."""
        has_xpu_time = any(avg.self_device_time_total > 0 for avg in averages)
        self.assertTrue(
            has_xpu_time,
            msg or "No events with positive self_device_time_total found",
        )

    @staticmethod
    def compute(x):
        x = x.to(device="xpu")
        return x + 1.0

    def test_correlation_id_mixed(self):
        input1 = torch.randn(3, 3, device="xpu")
        input2 = torch.randn(3, 3, device="xpu")

        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            output1 = input1 + 1.0
            output2 = input2 + 2.0
            output1 + output2

        # Both CPU and XPU events must be present
        self._assert_has_device_types(prof)

        # key_averages table must be non-empty
        averages = self._assert_key_averages_nonempty(prof)

        # At least one event should have positive XPU time
        self._assert_has_positive_xpu_time(averages)

        # Chrome trace must be well-formed and contain kernel events
        self._gen_and_check_json(prof, expected_cats=["kernel"])

        # Verify expected op names appear in profiler output
        event_names = {e.name for e in prof.events()}
        found_add = any("add" in name.lower() for name in event_names)
        self.assertTrue(
            found_add,
            f"Expected 'add' op in events. Found: {event_names}",
        )

    def test_profile_partial_runtime_ops(self):
        x = torch.randn(3, 3, device="cpu")

        # Warm up
        self.compute(x)

        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            output = self.compute(x)

        # Result must be on XPU
        self.assertEqual(output.device.type, "xpu")

        # Both CPU and XPU events must be present
        self._assert_has_device_types(prof)

        # key_averages table must be non-empty
        self._assert_key_averages_nonempty(prof)

        # Verify H2D transfer (aten::to or aten::_to_copy) appears in events
        event_names = {e.name for e in prof.events()}
        found_transfer = any(
            name in event_names
            for name in ("aten::to", "aten::_to_copy", "aten::copy_")
        )
        self.assertTrue(
            found_transfer,
            "Expected H2D transfer op (aten::to / aten::_to_copy / aten::copy_) "
            f"in events. Found: {event_names}",
        )

        # Verify add op appears
        found_add = any("add" in name.lower() for name in event_names)
        self.assertTrue(
            found_add,
            f"Expected 'add' op in events. Found: {event_names}",
        )

        # Chrome trace must be well-formed with non-negative durations
        count_names, _ = self._gen_and_check_json(prof)

        # At least some named events must be present in the trace
        self.assertGreater(len(count_names), 0, "No named events found in chrome trace")

    def test_xpu_kernel_time_present(self):
        shape = [4, 64, 128, 128]
        kernel_size = 2
        dtype = torch.float32

        pool = torch.nn.MaxPool2d(kernel_size, return_indices=True)
        unpool = torch.nn.MaxUnpool2d(kernel_size)

        cpu_input = torch.randn(shape, device="cpu", dtype=torch.float32)
        pooled, indices = pool(cpu_input)

        x_xpu = pooled.to(device="xpu", dtype=dtype)
        indices_xpu = indices.to(device="xpu", dtype=torch.int64)

        # Warm up
        for _ in range(3):
            x_tmp = x_xpu.clone().requires_grad_(True)
            y_tmp = unpool(x_tmp, indices_xpu, output_size=torch.Size(shape))
            grad = torch.randn(shape, device="xpu", dtype=dtype)
            y_tmp.backward(grad)

        # Profile forward + backward
        x_xpu_prof = x_xpu.clone().requires_grad_(True)
        grad_xpu = torch.randn(shape, device="xpu", dtype=dtype)

        with torch.profiler.profile(
            activities=self.ACTIVITIES,
        ) as prof:
            y_xpu = unpool(x_xpu_prof, indices_xpu, output_size=torch.Size(shape))
            y_xpu.backward(grad_xpu)

        # Both CPU and XPU events must be present
        self._assert_has_device_types(prof)

        # key_averages must be non-empty
        averages = self._assert_key_averages_nonempty(prof)

        # The original bug: XPU kernel times were missing/zero.
        # At least some events must have positive XPU time.
        xpu_times = [avg.self_device_time_total for avg in averages]
        self._assert_has_positive_xpu_time(
            averages,
            msg="No events with positive self_device_time_total — XPU kernel times "
            f"may be missing. XPU times: {xpu_times}",
        )

        # Chrome trace must contain kernel category events
        self._gen_and_check_json(prof, expected_cats=["kernel"])

    def test_kernel_set_stability(self):
        x = torch.randn(3, 3, device="cpu")

        # Warm up
        self.compute(x)

        iterations = 5
        first_kernel_names = None
        for i in range(iterations):
            with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
                self.compute(x)

            self._assert_has_device_types(
                prof, expected_types=[DeviceType.CPU, DeviceType.XPU]
            )

            # key_averages must be non-empty each iteration
            averages = prof.key_averages()
            self.assertGreater(
                len(averages),
                0,
                f"Iteration {i}: key_averages() returned no events",
            )

            # Collect kernel names from chrome trace for this iteration
            data = self._load_chrome_trace(prof)

            kernel_names = {
                e["name"]
                for e in data.get("traceEvents", [])
                if e.get("cat") == "kernel" and e.get("ph") == "X"
            }

            if i == 0:
                first_kernel_names = kernel_names
                self.assertGreater(
                    len(first_kernel_names),
                    0,
                    "Iteration 0: no kernel events found in chrome trace",
                )
            else:
                self.assertEqual(
                    kernel_names,
                    first_kernel_names,
                    f"Iteration {i}: kernel set differs from iteration 0. "
                    f"Added: {kernel_names - first_kernel_names}, "
                    f"Removed: {first_kernel_names - kernel_names}",
                )

    @unittest.skipUnless(has_triton(), "torch.compile with XPU backend requires triton")
    def test_triton_xpu_ops_profiling(self):
        @torch.compile
        def fn(x):
            x = x + 1.0
            x = x * x
            x = x + 2.0
            return x

        inp = torch.randn(128, 128, device="xpu")
        # Warm up (triggers compilation)
        fn(inp)

        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            output = fn(inp)

        # Result must be on XPU and have correct shape
        self.assertEqual(output.device.type, "xpu")
        self.assertEqual(output.shape, inp.shape)

        # Both device types must be present
        self._assert_has_device_types(prof)

        # key_averages must be non-empty
        self._assert_key_averages_nonempty(prof)

        # At least one kernel event should appear in the trace
        count_names, _ = self._gen_and_check_json(prof)
        self.assertGreater(len(count_names), 0, "No named events found in chrome trace")

    def test_event_duration_positive(self):
        a = torch.randn(64, 64, device="xpu")
        b = torch.randn(64, 64, device="xpu")

        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            c = torch.matmul(a, b)
            d = torch.relu(c)
            d + 1.0

        # XPU events must be present
        self._assert_has_device_types(prof)

        for event in prof.events():
            # cpu_time_total, self_cpu_time_total, and self_device_time_total
            # (XPU time) should all be non-negative
            self.assertGreaterEqual(
                event.cpu_time_total,
                0,
                f"Event '{event.name}' has negative cpu_time_total: "
                f"{event.cpu_time_total}",
            )
            self.assertGreaterEqual(
                event.self_cpu_time_total,
                0,
                f"Event '{event.name}' has negative self_cpu_time_total: "
                f"{event.self_cpu_time_total}",
            )
            self.assertGreaterEqual(
                event.self_device_time_total,
                0,
                f"Event '{event.name}' has negative self_device_time_total "
                f"(XPU time): {event.self_device_time_total}",
            )

    def test_chrome_trace_json_structure(self):
        a = torch.randn(32, 32, device="xpu")

        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            b = a + a
            c = b * 2.0
            c.sum()

        # _gen_and_check_json validates traceEvents structure, ph/name/cat/dur/ts
        # fields and asserts dur >= 0 and ts >= 0 for every X event.
        count_names, _ = self._gen_and_check_json(prof)

        # There should be at least some complete ("X") events
        self.assertGreater(
            sum(count_names.values()), 0, "No complete (ph=X) events in trace"
        )

        # XPU events must be present
        self._assert_has_device_types(prof)

    def test_profiler_with_record_function(self):
        a = torch.randn(16, 16, device="xpu")
        custom_label = "my_custom_region"

        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            with torch.profiler.record_function(custom_label):
                b = a + a
                b.sum()

        # XPU events must be present
        self._assert_has_device_types(prof)

        event_names = {e.name for e in prof.events()}
        self.assertIn(
            custom_label,
            event_names,
            f"Custom label '{custom_label}' not found in events. "
            f"Found: {event_names}",
        )

    def test_profiler_schedule_callback(self):
        wait, warmup, active = 1, 1, 2
        total_steps = wait + warmup + active

        traces = []

        def trace_handler(p):
            traces.append(list(p.events()))

        with torch.profiler.profile(
            activities=self.ACTIVITIES,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
        ) as prof:
            for _ in range(total_steps):
                a = torch.randn(8, 8, device="xpu")
                a + a
                prof.step()

        # The trace_handler should have been called once (after active phase)
        self.assertEqual(
            len(traces),
            1,
            f"Expected 1 trace callback, got {len(traces)}",
        )
        # The collected trace should have events
        self.assertGreater(len(traces[0]), 0, "Trace callback had no events")

        # XPU events must be present in the collected trace
        device_types = {e.device_type for e in traces[0]}
        self.assertIn(
            DeviceType.XPU,
            device_types,
            f"XPU events missing from schedule callback trace. Found: {device_types}",
        )

    def test_memory_events_present(self):
        with torch.profiler.profile(
            activities=self.ACTIVITIES,
            profile_memory=True,
        ) as prof:
            a = torch.randn(256, 256, device="xpu")
            b = a + a
            del a
            del b

        # Look for memory events in key_averages
        table_str = prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=-1
        )
        # The table should be produced without error
        self.assertGreater(len(table_str), 0)

        # Check chrome trace for memory events
        data = self._load_chrome_trace(prof)
        # Memory events use "i" (instant) phase with "[memory]" name
        memory_events = [
            e for e in data.get("traceEvents", []) if e.get("name") == "[memory]"
        ]
        self.assertGreater(
            len(memory_events),
            0,
            "No [memory] events found in trace with profile_memory=True",
        )

        # XPU events must be present
        self._assert_has_device_types(prof)

    def test_multi_op_kernel_to_op_association(self):
        a = torch.randn(100, 200, device="xpu")
        b = torch.randn(200, 300, device="xpu")

        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            c = torch.matmul(a, b)
            d = torch.add(c, 1.0)
            torch.relu(d)

        event_names = {e.name for e in prof.events()}

        # All three ops should appear
        found_mm = any("mm" in n.lower() or "matmul" in n.lower() for n in event_names)
        found_add = any("add" in n.lower() for n in event_names)
        found_relu = any("relu" in n.lower() for n in event_names)

        self.assertTrue(found_mm, f"matmul/mm not found in events: {event_names}")
        self.assertTrue(found_add, f"add not found in events: {event_names}")
        self.assertTrue(found_relu, f"relu not found in events: {event_names}")

        # Chrome trace should have kernel events with correlation IDs
        data = self._load_chrome_trace(prof)
        kernel_events = [
            e
            for e in data.get("traceEvents", [])
            if e.get("cat") == "kernel" and e.get("ph") == "X"
        ]
        self.assertGreater(
            len(kernel_events),
            0,
            "No kernel events found in chrome trace for multi-op workload",
        )

        # Each kernel event should have an "args" dict with correlation info
        for ke in kernel_events:
            self.assertIn(
                "args",
                ke,
                f"Kernel event '{ke.get('name')}' has no 'args' field",
            )

    def test_profiler_repeated_start_stop(self):
        for i in range(5):
            with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
                a = torch.randn(16, 16, device="xpu")
                a * a

            # Each run must produce some events
            self.assertGreater(
                len(prof.events()),
                0,
                f"Run {i}: profiler produced no events",
            )

            # XPU events must be present each run
            self._assert_has_device_types(prof, expected_types=[DeviceType.XPU])

    def test_backward_kernels_profiled(self):
        linear = torch.nn.Linear(64, 32).to("xpu")
        x = torch.randn(8, 64, device="xpu", requires_grad=True)

        # Warm up
        out = linear(x)
        out.sum().backward()

        x2 = torch.randn(8, 64, device="xpu", requires_grad=True)
        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            out = linear(x2)
            out.sum().backward()

        self._assert_has_device_types(prof)

        event_names = {e.name for e in prof.events()}

        # Forward ops should appear
        found_forward = any(
            "mm" in n.lower() or "linear" in n.lower() or "addmm" in n.lower()
            for n in event_names
        )
        self.assertTrue(
            found_forward,
            f"No forward pass ops (mm/linear/addmm) found: {event_names}",
        )

        # Backward ops should appear (autograd-related names)
        found_backward = any(
            "backward" in n.lower()
            or "autograd" in n.lower()
            or "MmBackward" in n
            or "AddmmBackward" in n
            for n in event_names
        )
        self.assertTrue(
            found_backward,
            f"No backward pass ops found: {event_names}",
        )

        # XPU time should be positive for at least some events
        self._assert_has_positive_xpu_time(
            prof.key_averages(),
            msg="No events with positive self_device_time_total during backward pass",
        )

    def test_h2d_d2h_transfer_events(self):
        cpu_tensor = torch.randn(64, 64, device="cpu")

        # Warm up
        cpu_tensor.to("xpu").to("cpu")

        with torch.profiler.profile(activities=self.ACTIVITIES) as prof:
            # H2D
            xpu_tensor = cpu_tensor.to("xpu")
            # Compute on XPU
            result = xpu_tensor + 1.0
            # D2H
            result.to("cpu")

        event_names = {e.name for e in prof.events()}

        # Transfer ops should appear
        transfer_ops = {"aten::to", "aten::_to_copy", "aten::copy_"}
        found_transfer = bool(transfer_ops & event_names)
        self.assertTrue(
            found_transfer,
            "No transfer ops (aten::to / aten::_to_copy / aten::copy_) "
            f"found in events: {event_names}",
        )

        # Both CPU and XPU device types must be present
        self._assert_has_device_types(prof)

        # Chrome trace should reflect transfers
        count_names, _ = self._gen_and_check_json(prof)
        self.assertGreater(len(count_names), 0, "No named events in chrome trace")


if __name__ == "__main__":
    run_tests()
