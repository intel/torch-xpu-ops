# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""Shared helpers for the pytest wrappers around the profiling repro scripts.

Device time is taken from the exported chrome trace rather than from
``key_averages().self_device_time_total``: the latter can report 0 on XPU and
mis-attributes time when ops are nested.
"""

import contextlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections import defaultdict

from torch.profiler import DeviceType

_HERE = os.path.dirname(os.path.abspath(__file__))

RUNTIME_OP_RE = re.compile(r"^ur[A-Z]")
LEVEL_ZERO_OP_RE = re.compile(r"^ze[A-Z]")
KERNEL_CATEGORIES = ("kernel", "gpu_kernel", "xpu_kernel")
DEVICE_ACTIVITY_CATEGORIES = KERNEL_CATEGORIES + ("gpu_memcpy", "gpu_memset")
CPU_OP_CATEGORIES = ("cpu_op", "user_annotation")

_DURATION_UNITS = {"us": 1.0, "ms": 1e3, "s": 1e6}
_DURATION_RE = re.compile(r"^([-+]?\d*\.?\d+)(us|ms|s)$")
_SEPARATOR_RE = re.compile(r"^-[- ]*$")


def load_script_module(filename):
    """Import a profiling script that lives next to this file.

    Uses importlib because ``reproducer.missing.gpu.kernel.time.py`` has dots
    in its name and cannot be imported with a plain ``import`` statement.
    """
    mod_name = "profiling_script_" + filename[: -len(".py")].replace(".", "_")
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_HERE, filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[mod_name]
        raise
    return module


def export_trace(prof):
    """Export the chrome trace of ``prof`` and return its ``traceEvents``."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    try:
        prof.export_chrome_trace(tmp.name)
        with open(tmp.name) as f:
            return json.load(f).get("traceEvents", [])
    finally:
        os.unlink(tmp.name)


@contextlib.contextmanager
def suppress_native_stderr():
    """Silence stderr at the file descriptor level.

    libkineto writes a ``USDT ... profiler_start/profiler_stop`` pair straight
    from C++ for every profiling window, which Python level redirection cannot
    capture.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def quiet_iterations(iterable, keep_first=1):
    """Yield from ``iterable``, muting native stderr while it produces the items
    after the first ``keep_first`` ones. Only the producer is muted, so anything
    the caller writes per iteration is still visible.
    """
    iterator = iter(iterable)
    index = 0
    while True:
        muted = (
            contextlib.nullcontext() if index < keep_first else suppress_native_stderr()
        )
        with muted:
            try:
                item = next(iterator)
            except StopIteration:
                return
        yield item
        index += 1


def _external_id(event):
    args = event.get("args") or {}
    for key in ("External id", "External Id", "external id"):
        if key in args:
            return args[key]
    return None


def kernel_events(events):
    """Return the complete device kernel events of a chrome trace."""
    return [
        e for e in events if e.get("ph") == "X" and e.get("cat") in KERNEL_CATEGORIES
    ]


def kernel_summary(events):
    """Return ``{kernel name: (call count, total duration in us)}``."""
    summary = defaultdict(lambda: [0, 0.0])
    for event in kernel_events(events):
        entry = summary[event.get("name", "unknown")]
        entry[0] += 1
        entry[1] += event.get("dur", 0)
    return {name: (count, total) for name, (count, total) in summary.items()}


def device_time_by_op(events):
    """Return ``{cpu op name: total device time in us}``.

    A kernel carries the ``External id`` of the op that launched it. The
    enclosing ops are recovered from the CPU timeline so that time is
    attributed to ancestors too, matching the ``XPU total`` column semantics.
    """
    ancestors_by_external_id = {}
    by_thread = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in CPU_OP_CATEGORIES:
            by_thread[(event.get("pid"), event.get("tid"))].append(event)

    for thread_events in by_thread.values():
        thread_events.sort(key=lambda e: (e.get("ts", 0), -e.get("dur", 0)))
        open_ops = []
        for event in thread_events:
            start = event.get("ts", 0)
            while open_ops and open_ops[-1][0] <= start:
                open_ops.pop()
            name = event.get("name", "unknown")
            external_id = _external_id(event)
            if external_id is not None:
                ancestors_by_external_id[external_id] = [n for _, n in open_ops] + [
                    name
                ]
            open_ops.append((start + event.get("dur", 0), name))

    totals = defaultdict(float)
    for kernel in kernel_events(events):
        chain = ancestors_by_external_id.get(_external_id(kernel), [])
        # dict.fromkeys dedups recursive same-name ops without double counting
        for name in dict.fromkeys(chain):
            totals[name] += kernel.get("dur", 0)
    return dict(totals)


def parse_profiler_time(cell):
    """Convert one duration cell of a profiler table into microseconds."""
    match = _DURATION_RE.match(cell.strip())
    if match is None:
        raise ValueError(f"unexpected duration cell: {cell!r}")
    return float(match.group(1)) * _DURATION_UNITS[match.group(2)]


def parse_profiler_table(table):
    """Parse the text of ``key_averages().table()`` into a list of dict rows.

    Needed to check a table that was saved as text (rn50 stores its table with
    ``torch.save``). Op names may contain spaces, so the name column is sliced
    at the width given by the dashed separator; the other cells never do and
    are split on whitespace.
    """
    lines = table.splitlines()
    separators = [i for i, line in enumerate(lines) if _SEPARATOR_RE.match(line)]
    if len(separators) < 3:
        raise ValueError("no profiler table found in the given text")
    spans = [(m.start(), m.end()) for m in re.finditer(r"-+", lines[separators[0]])]
    columns = [lines[separators[0] + 1][start:end].strip() for start, end in spans]
    name_width = spans[0][1]

    rows = []
    for line in lines[separators[1] + 1 : separators[2]]:
        if not line.strip():
            continue
        cells = line[name_width:].split()
        if len(cells) != len(columns) - 1:
            raise ValueError(f"cannot parse profiler table row: {line!r}")
        row = dict(zip(columns[1:], cells))
        row[columns[0]] = line[:name_width].strip()
        rows.append(row)
    return rows


def runtime_op_averages(prof, pattern=RUNTIME_OP_RE):
    """Return the ``key_averages()`` rows whose name matches ``pattern``."""
    return [avg for avg in prof.key_averages() if pattern.match(avg.key)]


def _format_runtime_row(avg):
    return (
        f"  {avg.key:<58.58s} calls={avg.count:<6d} "
        f"cpu_total={avg.cpu_time_total:12.3f}us "
        f"xpu_total={avg.device_time_total:12.3f}us"
    )


def dump_profile_report(prof, case_name, sort_by="self_xpu_time_total", events=None):
    """Print the human readable profiling report for manual cross checking.

    Keeps the same table (and ``sort_by``) the standalone script used to print,
    then adds the numbers the assertions are based on.
    """
    if events is None:
        events = export_trace(prof)

    print(f"\n===== [{case_name}] profiler table =====")
    print(prof.key_averages().table(sort_by=sort_by, row_limit=-1))

    print(f"----- [{case_name}] trace kernel summary -----")
    summary = kernel_summary(events)
    if summary:
        for name, (count, total) in sorted(summary.items(), key=lambda kv: -kv[1][1]):
            print(
                f"  calls={count:<6d} total={total:12.3f}us "
                f"avg={total / count:12.3f}us  {name}"
            )
    else:
        print("  <no kernel events in trace>")

    for title, pattern in (
        ("runtime ops (ur*)", RUNTIME_OP_RE),
        ("runtime ops (ze*) [observe only]", LEVEL_ZERO_OP_RE),
    ):
        print(f"----- [{case_name}] {title} -----")
        rows = runtime_op_averages(prof, pattern)
        if rows:
            for avg in rows:
                print(_format_runtime_row(avg))
        else:
            print("  <none>")

    # Only device-side rows are summed: a CPU op row repeats the time of the
    # kernel rows below it, so summing everything would double count.
    key_averages_total = sum(
        avg.self_device_time_total
        for avg in prof.key_averages()
        if avg.device_type != DeviceType.CPU
    )
    trace_total = sum(
        event.get("dur", 0)
        for event in events
        if event.get("ph") == "X" and event.get("cat") in DEVICE_ACTIVITY_CATEGORIES
    )
    delta = key_averages_total - trace_total
    print(f"----- [{case_name}] cross check -----")
    print(f"  sum(key_averages device rows) = {key_averages_total:12.3f}us")
    print(f"  sum(trace device activity)    = {trace_total:12.3f}us")
    print(f"  delta                         = {delta:12.3f}us")
    return events


def assert_common(tc, prof, case_name, sort_by="self_xpu_time_total", events=None):
    """Print the report, then assert the criteria every profiling case shares.

    G1: the trace contains at least one device kernel with a positive duration.
    G2: every runtime op (``ur*``) has XPU time 0 and a positive CPU time.
    ``ze*`` rows are only reported, not asserted, until enough data is gathered.
    """
    events = dump_profile_report(prof, case_name, sort_by=sort_by, events=events)

    kernels = kernel_events(events)
    tc.assertTrue(
        any(k.get("dur", 0) > 0 for k in kernels),
        f"[{case_name}] G1 failed: no device kernel with positive duration in the "
        f"chrome trace ({len(kernels)} kernel events found)",
    )

    runtime_rows = runtime_op_averages(prof)
    tc.assertTrue(
        runtime_rows,
        f"[{case_name}] G2 failed: the profiler reported no runtime op (ur*)",
    )
    for avg in runtime_rows:
        tc.assertEqual(
            avg.device_time_total,
            0,
            f"[{case_name}] G2 failed: runtime op must have no XPU time\n"
            f"{_format_runtime_row(avg)}",
        )
        tc.assertGreater(
            avg.cpu_time_total,
            0,
            f"[{case_name}] G2 failed: runtime op must have CPU time\n"
            f"{_format_runtime_row(avg)}",
        )
    return events
