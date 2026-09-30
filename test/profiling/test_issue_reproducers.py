# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""Pass/fail wrappers around the standalone profiling issue-reproduce scripts.

Each test runs the very same workload as the script it wraps, prints the same
profiler table for manual cross checking, and then asserts the criteria that
used to be verified by eyeballing the log.
"""

import os
import sys
import unittest

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._triton import has_triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profiling_test_utils import (
    assert_common,
    device_time_by_op,
    dump_profile_report,
    export_trace,
    kernel_summary,
    load_script_module,
    quiet_iterations,
    runtime_op_averages,
)


class TestProfilingIssueReproducers(TestCase):
    def test_correlation_id_mixed(self):
        script = load_script_module("correlation_id_mixed.py")
        prof = script.run_profile()
        events = assert_common(
            self, prof, "correlation_id_mixed", sort_by=script.SORT_BY
        )

        device_time = device_time_by_op(events)
        timed_aten_ops = {
            name: total
            for name, total in device_time.items()
            if name.startswith("aten::") and total > 0
        }
        self.assertTrue(
            timed_aten_ops,
            "No aten:: op was attributed any XPU time. "
            f"Device time per op: {device_time}",
        )

    def test_missing_gpu_kernel_time(self):
        script = load_script_module("reproducer.missing.gpu.kernel.time.py")
        prof = script.run_profile()
        events = assert_common(
            self,
            prof,
            "reproducer.missing.gpu.kernel.time",
            sort_by=script.SORT_BY,
        )

        device_time = device_time_by_op(events)
        self.assertGreater(
            device_time.get("aten::gather", 0),
            0,
            "aten::gather has no XPU time — the missing kernel time issue is back. "
            f"Device time per op: {device_time}",
        )
        timed_backward_ops = {
            name: total
            for name, total in device_time.items()
            if "backward" in name.lower() and total > 0
        }
        self.assertTrue(
            timed_backward_ops,
            "No backward op was attributed any XPU time. "
            f"Device time per op: {device_time}",
        )

    def test_time_precision(self):
        script = load_script_module("time_precision_in_profile.py")
        iters = int(
            os.environ.get("PROFILING_TIME_PRECISION_ITERS", script.DEFAULT_ITERS)
        )

        anomalies = []
        last = None
        # Only the first window keeps its native stderr: libkineto logs a
        # profiler_start/stop pair per window and `iters` copies is pure noise.
        for i, prof in quiet_iterations(script.run_profile(iters)):
            events = export_trace(prof)
            negative_averages = [
                avg
                for avg in prof.key_averages()
                if avg.cpu_time_total < 0
                or avg.self_cpu_time_total < 0
                or avg.device_time_total < 0
            ]
            negative_events = [
                e for e in events if e.get("ph") == "X" and e.get("dur", 0) < 0
            ]
            if negative_averages or negative_events:
                anomalies.append(
                    f"iter={i}: "
                    f"averages={[avg.key for avg in negative_averages]}, "
                    f"trace events={[e.get('name') for e in negative_events]}"
                )
                # Only anomalous iterations are dumped in full; 1000 tables would
                # bury the failure.
                dump_profile_report(
                    prof,
                    f"time_precision_in_profile[iter={i}] ANOMALY",
                    sort_by=script.SORT_BY,
                    events=events,
                )

            if i == 0:
                assert_common(
                    self,
                    prof,
                    "time_precision_in_profile[iter=0]",
                    sort_by=script.SORT_BY,
                    events=events,
                )
            last = (i, prof, events)

        self.assertIsNotNone(last, "time_precision_in_profile ran zero iterations")
        last_i, last_prof, last_events = last
        if last_i != 0:
            assert_common(
                self,
                last_prof,
                f"time_precision_in_profile[iter={last_i}]",
                sort_by=script.SORT_BY,
                events=last_events,
            )

        self.assertEqual(
            anomalies,
            [],
            f"Negative durations found in {len(anomalies)}/{iters} iterations:\n"
            + "\n".join(anomalies),
        )

    def test_partial_runtime_ops(self):
        script = load_script_module("profile_partial_runtime_ops.py")
        prof = script.run_profile()
        assert_common(self, prof, "profile_partial_runtime_ops", sort_by=script.SORT_BY)

        reported = {avg.key for avg in runtime_op_averages(prof)}
        expected = set(script.EXPECTED_RUNTIME_OPS)
        self.assertEqual(
            reported,
            expected,
            f"Unexpected runtime ops: {sorted(reported - expected)}; "
            f"missing runtime ops: {sorted(expected - reported)}",
        )

    @unittest.skipUnless(has_triton(), "torch.compile with XPU backend requires triton")
    def test_triton_xpu_ops_time(self):
        script = load_script_module("triton_xpu_ops_time.py")
        prof = script.run_profile()
        events = assert_common(
            self, prof, "triton_xpu_ops_time", sort_by=script.SORT_BY
        )

        summary = kernel_summary(events)
        triton_kernels = {
            name: value for name, value in summary.items() if "triton" in name.lower()
        }
        self.assertTrue(
            triton_kernels,
            f"No triton kernel found in the trace. Kernels: {sorted(summary)}",
        )
        for name, (count, total) in triton_kernels.items():
            self.assertGreater(
                total,
                0,
                f"Triton kernel '{name}' has no XPU time ({count} calls)",
            )


if __name__ == "__main__":
    run_tests()
