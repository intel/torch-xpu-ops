"""
pytest conftest for Intel XPU runs.

Features
--------
- Empties XPU cache and resets allocator stats before every test (autouse fixture).
- Records per-test peak/leak memory into the JUnit XML (<properties> on <testcase>).
- Detects crash / OOM patterns in test output and force-restarts the xdist worker.
- Persists per-test records to disk (JSONL, fsync'd) so results survive a worker kill.
- Prints a terminal summary with status counts, total time, and a memory-peak
  distribution bucketed at 50/60/70/80/90/100%/100+(OOM).

Run
---
    pytest -n 4 --max-worker-restart=20 \
           --junit-xml=report.xml -o junit_family=xunit2

Environment variables
---------------------
    XPU_RESULTS_DIR    Directory for JSONL results (default: $TMPDIR/xpu_pytest_results)
    XPU_CLEAN_RESULTS  "1" (default) wipes JSONL files after summary; "0" keeps them.
    XPU_REPORT_CSV     Output path for the per-test CSV (default: xpu_report.csv).
    XPU_REPORT_HTML    Output path for the HTML report (default: xpu_report.html).
"""

from __future__ import annotations

import csv
import html as _html

import atexit
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import pytest
import torch


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
_GiB = 1024 ** 3

_CRASH_PATTERNS = [
    re.compile(r"ur_result_error",    re.IGNORECASE),
    re.compile(r"segmentation fault", re.IGNORECASE),
    re.compile(r"bus error",          re.IGNORECASE),
    re.compile(r"kernel died",        re.IGNORECASE),
    re.compile(r"illegal memory",     re.IGNORECASE),
    re.compile(r"out.*of.*memory",    re.IGNORECASE),
]
# Single combined regex used on the hot path (one search vs. six).
_CRASH_RE = re.compile(
    "|".join(f"(?:{p.pattern})" for p in _CRASH_PATTERNS),
    re.IGNORECASE,
)
_OOM_PATTERN = re.compile(r"out.*of.*memory", re.IGNORECASE)

# Allow disabling empty_cache() for users who want max throughput
# (allocator stays warm across tests, but cross-test isolation is weaker).
_DO_EMPTY_CACHE = os.environ.get("XPU_EMPTY_CACHE", "1") != "0"

# Memory bucket upper bounds, in percent of device total memory.
# A test lands in the first bucket whose upper bound it does NOT exceed.
_MEM_BUCKETS = [50, 60, 70, 80, 90, 100]

# Non-zero exit code so pytest-xdist treats the worker as crashed and restarts it.
_WORKER_RESTART_EXITCODE = 70


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


# Cache device totals; they never change during a run and the property
# lookup hits the driver.
_DEVICE_TOTAL_CACHE: dict[int, int] = {}


def _device_total_bytes(dev) -> int:
    key = int(dev) if isinstance(dev, int) else id(dev)
    v = _DEVICE_TOTAL_CACHE.get(key)
    if v is not None:
        return v
    try:
        v = int(torch.xpu.get_device_properties(dev).total_memory)
    except Exception:
        v = 0
    _DEVICE_TOTAL_CACHE[key] = v
    return v


def _collect_device_info() -> list[dict]:
    """Best-effort snapshot of available XPU devices."""
    info: list[dict] = []
    if not _xpu_available():
        return info
    try:
        n = torch.xpu.device_count()
    except Exception:
        n = 0
    for i in range(n):
        d: dict = {"index": i}
        try:
            p = torch.xpu.get_device_properties(i)
            d["name"] = getattr(p, "name", None)
            d["total_memory_B"] = int(getattr(p, "total_memory", 0))
            for attr in (
                "driver_version", "device_id", "vendor_id", "platform_name",
                "max_compute_units", "gpu_eu_count", "gpu_subslice_count",
                "max_work_group_size", "sub_group_sizes", "type",
            ):
                v = getattr(p, attr, None)
                if v is not None:
                    d[attr] = v if isinstance(v, (int, float, str, bool, list, tuple)) else str(v)
        except Exception as e:
            d["error"] = repr(e)
        info.append(d)
    return info


def _format_bytes_g(n: int) -> str:
    return f"{n / _GiB:.2f} GiB"


def _collect_software_info() -> dict:
    """Environment versions worth knowing for reproducibility."""
    import platform
    import shutil
    import subprocess
    import sys

    info: dict = {
        "os":       "",
        "platform": platform.platform(),
        "driver":   "",
        "gcc":      "",
        "python":   sys.version.split()[0],
        "torch":    getattr(torch, "__version__", "?"),
        "torchvision": "",
        "triton":   "",
    }

    # OS pretty name (Linux) or platform.system() fallback.
    try:
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                kv = {}
                for line in f:
                    if "=" in line:
                        k, v = line.rstrip().split("=", 1)
                        kv[k] = v.strip().strip('"')
            info["os"] = kv.get("PRETTY_NAME") or kv.get("NAME") or platform.system()
        else:
            info["os"] = platform.system()
    except Exception:
        info["os"] = platform.system()

    # gcc version (best effort).
    try:
        gcc = shutil.which("gcc")
        if gcc:
            out = subprocess.check_output([gcc, "-dumpfullversion", "-dumpversion"],
                                          stderr=subprocess.DEVNULL, text=True).strip()
            info["gcc"] = out.splitlines()[0]
    except Exception:
        pass

    # torchvision / triton versions.
    for mod in ("torchvision", "triton"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "present")
        except Exception:
            info[mod] = ""

    return info


def _matches(text: str, patterns) -> str | None:
    if not text:
        return None
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


def _abort_worker(reason: str) -> None:
    """Hard-exit so pytest-xdist treats this worker as crashed and restarts it."""
    try:
        os.write(2, f"\n[conftest] aborting worker (pid={os.getpid()}): {reason}\n".encode())
    except Exception:
        pass
    # os._exit bypasses pytest/atexit handlers, which is required to look like a crash.
    os._exit(_WORKER_RESTART_EXITCODE)


def _bucket_label(pct: float, oom: bool) -> str:
    if oom or pct > 100:
        return "100+ (OOM)"
    prev = 0
    for ub in _MEM_BUCKETS:
        if pct <= ub:
            return f"{prev:>3d}-{ub:<3d}%"
        prev = ub
    return "100+ (OOM)"


# --------------------------------------------------------------------------- #
# Per-process state: append-only JSONL + in-memory pending record
# --------------------------------------------------------------------------- #
_RESULTS_DIR = Path(os.environ.get("XPU_RESULTS_DIR", tempfile.gettempdir())) / "xpu_pytest_results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "main")
_RESULTS_FILE = _RESULTS_DIR / f"results-{os.getpid()}-{_WORKER_ID}.jsonl"
_RESULTS_FP = open(_RESULTS_FILE, "a", buffering=1)  # line-buffered
atexit.register(lambda: _RESULTS_FP.close())

# Drop a device-info snapshot once per process (controller reads any of them).
_DEVICE_INFO_FILE = _RESULTS_DIR / f"device-{os.getpid()}-{_WORKER_ID}.json"
try:
    _info = _collect_device_info()
    if _info:
        _DEVICE_INFO_FILE.write_text(json.dumps({
            "pid": os.getpid(),
            "worker": _WORKER_ID,
            "torch_version": getattr(torch, "__version__", "?"),
            "devices": _info,
        }))
except Exception:
    pass

_PENDING: dict[str, dict] = {}


def _flush(nodeid: str, *, durable: bool = False) -> None:
    """Append a record to the JSONL file.

    durable=True forces fsync (needed before we hard-exit the worker so the
    record survives os._exit / SIGKILL). The default path skips fsync to
    avoid 1-10ms of disk-sync overhead per test.
    """
    rec = _PENDING.pop(nodeid, None)
    if rec is None:
        return
    rec["nodeid"] = nodeid
    try:
        _RESULTS_FP.write(json.dumps(rec) + "\n")
        _RESULTS_FP.flush()
        if durable:
            os.fsync(_RESULTS_FP.fileno())
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Autouse fixture: empty cache + record peak
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def xpu_mem_tracker(request, record_property):
    if not _xpu_available():
        yield
        return

    dev = torch.xpu.current_device()
    total = _device_total_bytes(dev)

    # empty_cache BEFORE the test (not after) so the allocator stays warm.
    # Opt out with XPU_EMPTY_CACHE=0 for max throughput.
    if _DO_EMPTY_CACHE:
        torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats(dev)
    before = torch.xpu.memory_allocated(dev)

    yield

    after         = torch.xpu.memory_allocated(dev)
    peak_alloc    = torch.xpu.max_memory_allocated(dev)
    peak_reserved = torch.xpu.max_memory_reserved(dev)
    peak_pct      = (peak_alloc / total * 100.0) if total else 0.0

    record_property("xpu_alloc_before_GiB",  round(before        / _GiB, 6))
    record_property("xpu_alloc_after_GiB",   round(after         / _GiB, 6))
    record_property("xpu_peak_alloc_GiB",    round(peak_alloc    / _GiB, 6))
    record_property("xpu_peak_reserved_GiB", round(peak_reserved / _GiB, 6))
    record_property("xpu_peak_alloc_pct",    round(peak_pct, 3))
    record_property("xpu_leak_GiB",          round((after - before) / _GiB, 6))

    rec = _PENDING.setdefault(request.node.nodeid, {})
    rec["peak_pct"]     = peak_pct
    rec["peak_alloc_B"] = int(peak_alloc)


# --------------------------------------------------------------------------- #
# Hook: collect status + duration; detect crash/OOM patterns
# --------------------------------------------------------------------------- #
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    rec = _PENDING.setdefault(item.nodeid, {})

    # Pick up the test's overall outcome. Prefer the 'call' phase; fall back to
    # 'setup' if the test errored before the body ran.
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        rec["status"]   = report.outcome  # passed / failed / skipped
        rec["duration"] = float(getattr(report, "duration", 0.0))

    # Hot path optimization: only scan for crash/OOM patterns when there is
    # something worth scanning. A passing test with no captured output skips
    # all regex work.
    needs_scan = (
        report.failed
        or call.excinfo is not None
        or bool(getattr(report, "sections", None))
    )

    if needs_scan:
        haystacks = []
        if report.failed and report.longrepr:
            haystacks.append(str(report.longrepr))
        for _title, content in getattr(report, "sections", []) or []:
            if content:
                haystacks.append(content)
        if call.excinfo is not None:
            haystacks.append(str(call.excinfo.value))
        blob = "\n".join(haystacks)

        if blob:
            if _OOM_PATTERN.search(blob):
                rec["oom"] = True
            m = _CRASH_RE.search(blob)
            if m:
                hit = m.group(0)
                # Force durable write before killing the worker so we don't
                # lose the in-flight test record across SIGKILL.
                _flush(item.nodeid, durable=True)
                _abort_worker(
                    f"matched crash pattern {hit!r} in {report.nodeid} ({report.when})"
                )

    # Last phase for this test -> persist record (no fsync; normal exit path).
    if report.when == "teardown":
        _flush(item.nodeid)


# --------------------------------------------------------------------------- #
# Aggregation + terminal summary (controller side)
# --------------------------------------------------------------------------- #
def _load_all_results() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for f in sorted(_RESULTS_DIR.glob("results-*.jsonl")):
        try:
            with open(f) as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    nid = rec.pop("nodeid", None)
                    if nid:
                        # Last write wins (handles retries on restarted workers).
                        merged.setdefault(nid, {}).update(rec)
        except OSError:
            continue
    return merged


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "xpu_mem: tracked by the XPU memory fixture (autouse).",
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Only the controller prints the merged summary.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    results = _load_all_results()
    if not results:
        return

    tr = terminalreporter
    status_counts = Counter()
    bucket_counts = Counter()
    total_time = 0.0
    peak_max_pct = 0.0
    peak_max_node = ""
    crashed = 0

    for nodeid, rec in results.items():
        status = rec.get("status")
        if status is None:
            # Test started but never reached teardown -> worker died on it.
            status = "crashed"
            crashed += 1
        status_counts[status] += 1
        total_time += rec.get("duration", 0.0)

        pct = float(rec.get("peak_pct", 0.0))
        oom = bool(rec.get("oom", False))
        bucket_counts[_bucket_label(pct, oom)] += 1

        if pct > peak_max_pct:
            peak_max_pct = pct
            peak_max_node = nodeid

    tr.write_sep("=", "XPU summary")

    # ---- device info (hardware + cached snapshot fallback) -------------
    dev_info = None
    live = _collect_device_info()
    if live:
        dev_info = {"devices": live}
    else:
        for f in sorted(_RESULTS_DIR.glob("device-*.json")):
            try:
                dev_info = json.loads(f.read_text())
                break
            except Exception:
                continue

    # ===== Device info ==================================================
    tr.write_sep("-", "Device info")
    if dev_info and dev_info.get("devices"):
        for d in dev_info["devices"]:
            idx = d.get("index", "?")
            name = d.get("name", "?")
            total = d.get("total_memory_B", 0)
            tr.write_line(f"  xpu[{idx}] : {name}")
            tr.write_line(f"            memory = {_format_bytes_g(total)}")
    else:
        tr.write_line("  xpu: not available")

    # ===== Environment ==================================================
    tr.write_sep("-", "Environment")
    sw = _collect_software_info()
    # Pull driver version from device properties if available.
    if dev_info and dev_info.get("devices"):
        d0 = dev_info["devices"][0]
        if not sw.get("driver"):
            sw["driver"] = d0.get("driver_version", "")

    order = ["os", "platform", "driver", "gcc", "python", "torch", "torchvision", "triton"]
    key_width = max(len(k) for k in order)
    for k in order:
        tr.write_line(f"  {k:<{key_width}} : {sw.get(k, '')}")

    # ===== Test results =================================================
    tr.write_sep("-", "Test results")
    parts = [f"{k}={v}" for k, v in sorted(status_counts.items())]
    tr.write_line(f"  tests : {sum(status_counts.values())}  ({', '.join(parts)})")
    tr.write_line(f"  time  : {total_time:.2f}s total")

    tr.write_line("  memory peak distribution (peak_alloc / device_total):")
    ordered = [f"{prev:>3d}-{ub:<3d}%" for prev, ub in zip([0] + _MEM_BUCKETS[:-1], _MEM_BUCKETS)]
    ordered.append("100+ (OOM)")
    width = max(len(b) for b in ordered)
    for b in ordered:
        n = bucket_counts.get(b, 0)
        bar = "#" * min(n, 40)
        tr.write_line(f"    {b:<{width}} : {n:>5d}  {bar}")

    if peak_max_node:
        tr.write_line(f"  highest peak : {peak_max_pct:.2f}%  in {peak_max_node}")
    if crashed:
        tr.write_line(f"  crashed (no teardown) : {crashed}")

    # ===== Write CSV + HTML reports =====================================
    csv_path = Path(os.environ.get("XPU_REPORT_CSV", "xpu_report.csv"))
    html_path = Path(os.environ.get("XPU_REPORT_HTML", "xpu_report.html"))
    try:
        _write_csv_report(csv_path, results)
        tr.write_line(f"  csv  : {csv_path}")
    except Exception as e:
        tr.write_line(f"  csv  : FAILED ({e!r})")
    try:
        _write_html_report(
            html_path,
            results=results,
            status_counts=status_counts,
            bucket_counts=bucket_counts,
            total_time=total_time,
            peak_max_pct=peak_max_pct,
            peak_max_node=peak_max_node,
            crashed=crashed,
            dev_info=dev_info,
            sw=sw,
        )
        tr.write_line(f"  html : {html_path}")
    except Exception as e:
        tr.write_line(f"  html : FAILED ({e!r})")

    # Clean up JSONL files unless the user asked to keep them.
    if os.environ.get("XPU_CLEAN_RESULTS", "1") == "1":
        for f in _RESULTS_DIR.glob("results-*.jsonl"):
            try:
                f.unlink()
            except OSError:
                pass
        for f in _RESULTS_DIR.glob("device-*.json"):
            try:
                f.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# CSV + HTML report writers
# --------------------------------------------------------------------------- #
_CSV_COLUMNS = [
    "nodeid", "status", "duration_s",
    "peak_alloc_GiB", "peak_alloc_pct", "oom",
]


def _row_for(nodeid: str, rec: dict) -> dict:
    peak_B = float(rec.get("peak_alloc_B", 0))
    return {
        "nodeid":         nodeid,
        "status":         rec.get("status") or "crashed",
        "duration_s":     round(float(rec.get("duration", 0.0)), 4),
        "peak_alloc_GiB": round(peak_B / _GiB, 6),
        "peak_alloc_pct": round(float(rec.get("peak_pct", 0.0)), 3),
        "oom":            bool(rec.get("oom", False)),
    }


def _write_csv_report(path: Path, results: dict[str, dict]) -> None:
    rows = [_row_for(nid, rec) for nid, rec in results.items()]
    rows.sort(key=lambda r: (-r["peak_alloc_pct"], r["nodeid"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>XPU pytest report</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #111827;
    --panel-2: #1f2937;
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --good: #10b981;
    --warn: #f59e0b;
    --bad:  #ef4444;
    --skip: #64748b;
    --border: #1e293b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 32px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
    color: var(--text);
    min-height: 100vh;
  }
  h1 { margin: 0 0 4px; font-size: 28px; letter-spacing: -0.02em; }
  .subtitle { color: var(--muted); margin-bottom: 24px; font-size: 14px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.02), 0 8px 24px rgba(0,0,0,0.25);
  }
  .card h3 { margin: 0 0 12px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 600; }
  .kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; font-size: 14px; }
  .kv .k { color: var(--muted); }
  .kv .v { color: var(--text); word-break: break-word; }
  .stat-row { display: flex; flex-wrap: wrap; gap: 16px; }
  .stat {
    flex: 1 1 auto; min-width: 120px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }
  .stat .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .stat .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .pill.passed  { background: rgba(16,185,129,0.15); color: var(--good); }
  .pill.failed  { background: rgba(239,68,68,0.15);  color: var(--bad); }
  .pill.skipped { background: rgba(100,116,139,0.2); color: var(--skip); }
  .pill.crashed { background: rgba(245,158,11,0.18); color: var(--warn); }
  .bars { display: grid; gap: 8px; margin-top: 6px; }
  .bar-row { display: grid; grid-template-columns: 110px 1fr 60px; align-items: center; gap: 12px; font-size: 13px; }
  .bar { height: 10px; background: var(--panel-2); border-radius: 999px; overflow: hidden; }
  .bar > span { display: block; height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8); }
  .bar.oom > span { background: linear-gradient(90deg, #f59e0b, #ef4444); }
  .toolbar { display: flex; gap: 8px; margin: 8px 0 12px; flex-wrap: wrap; }
  .toolbar input, .toolbar select {
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 8px 10px; font-size: 13px;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead th {
    text-align: left; font-weight: 600; color: var(--muted);
    padding: 10px 12px; border-bottom: 1px solid var(--border);
    cursor: pointer; user-select: none; white-space: nowrap;
  }
  thead th:hover { color: var(--accent); }
  thead tr.filter-row th {
    padding: 6px 8px; cursor: default; font-weight: 400;
    background: rgba(255,255,255,0.015);
  }
  thead tr.filter-row th:hover { color: var(--muted); }
  .col-filter, .col-filter-min, .col-filter-max {
    width: 100%; min-width: 70px;
    background: var(--panel-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 5px 8px; font-size: 12px;
  }
  .col-filter-range { display: flex; gap: 4px; }
  .col-filter-range > input { flex: 1; min-width: 0; }
  tbody td { padding: 9px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); vertical-align: top; }
  tbody tr:hover { background: rgba(255,255,255,0.02); }
  td.nodeid { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; word-break: break-all; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .mem-cell { display: flex; align-items: center; gap: 8px; }
  .mem-cell .bar { width: 80px; }
  footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
</style>
</head>
<body>
  <h1>XPU pytest report</h1>
  <div class="subtitle">Generated {{generated_at}} &middot; {{n_tests}} tests &middot; {{total_time}}s</div>

  <div class="grid">
    <div class="card">
      <h3>Device</h3>
      <div class="kv">{{device_kv}}</div>
    </div>
    <div class="card">
      <h3>Environment</h3>
      <div class="kv">{{env_kv}}</div>
    </div>
    <div class="card">
      <h3>Status</h3>
      <div class="stat-row">{{status_stats}}</div>
    </div>
    <div class="card">
      <h3>Memory peak distribution</h3>
      <div class="bars">{{bucket_bars}}</div>
    </div>
  </div>

  <div class="card">
    <h3>Per-test results</h3>
    <div style="overflow:auto; max-height: 70vh;">
    <table id="results">
      <thead>
        <tr>
          <th data-sort="str">Test</th>
          <th data-sort="str">Status</th>
          <th data-sort="num">Time (s)</th>
          <th data-sort="num">Peak (GiB)</th>
          <th data-sort="num">Peak (%)</th>
          <th data-sort="str">OOM</th>
        </tr>
        <tr class="filter-row">
          <th><input class="col-filter" data-col="0" data-type="text" type="search" placeholder="contains&hellip;"></th>
          <th>
            <select class="col-filter" data-col="1" data-type="select">
              <option value="">all</option>
              <option value="passed">passed</option>
              <option value="failed">failed</option>
              <option value="skipped">skipped</option>
              <option value="crashed">crashed</option>
            </select>
          </th>
          <th><div class="col-filter-range">
            <input class="col-filter-min" data-col="2" data-type="num" data-bound="min" type="number" step="any" placeholder="min">
            <input class="col-filter-max" data-col="2" data-type="num" data-bound="max" type="number" step="any" placeholder="max">
          </div></th>
          <th><div class="col-filter-range">
            <input class="col-filter-min" data-col="3" data-type="num" data-bound="min" type="number" step="any" placeholder="min">
            <input class="col-filter-max" data-col="3" data-type="num" data-bound="max" type="number" step="any" placeholder="max">
          </div></th>
          <th><div class="col-filter-range">
            <input class="col-filter-min" data-col="4" data-type="num" data-bound="min" type="number" step="any" placeholder="min">
            <input class="col-filter-max" data-col="4" data-type="num" data-bound="max" type="number" step="any" placeholder="max">
          </div></th>
          <th>
            <select class="col-filter" data-col="5" data-type="select">
              <option value="">all</option>
              <option value="yes">yes</option>
              <option value="no">no</option>
            </select>
          </th>
        </tr>
      </thead>
      <tbody>{{rows}}</tbody>
    </table>
    </div>
  </div>

  <footer>conftest.py &middot; XPU memory tracker</footer>

<script>
(function() {
  const tbl = document.getElementById('results');
  const tbody = tbl.tBodies[0];
  const inputs = tbl.querySelectorAll('.col-filter, .col-filter-min, .col-filter-max');

  // Build filter spec from current inputs.
  function spec() {
    const cols = {};
    inputs.forEach(el => {
      const c = +el.dataset.col;
      const t = el.dataset.type;
      cols[c] = cols[c] || { type: t };
      if (t === 'text' || t === 'select') {
        cols[c].value = el.value.trim().toLowerCase();
      } else if (t === 'num') {
        const v = el.value === '' ? null : parseFloat(el.value);
        cols[c][el.dataset.bound] = (Number.isFinite(v) ? v : null);
      }
    });
    return cols;
  }

  function rowVal(tr, col) {
    const cell = tr.cells[col];
    return cell.dataset.v ?? cell.textContent;
  }

  function apply() {
    const cols = spec();
    for (const tr of tbody.rows) {
      let visible = true;
      for (const c in cols) {
        const f = cols[c];
        const raw = rowVal(tr, +c);
        if (f.type === 'text') {
          if (f.value && !String(raw).toLowerCase().includes(f.value)) { visible = false; break; }
        } else if (f.type === 'select') {
          if (!f.value) continue;
          // Status is read from dataset on the row; OOM column text is "yes" or "".
          let target;
          if (+c === 1) target = (tr.dataset.status || '').toLowerCase();
          else if (+c === 5) target = (String(raw).trim().toLowerCase() === 'yes') ? 'yes' : 'no';
          else target = String(raw).toLowerCase();
          if (target !== f.value) { visible = false; break; }
        } else if (f.type === 'num') {
          const num = parseFloat(raw);
          if (f.min != null && !(num >= f.min)) { visible = false; break; }
          if (f.max != null && !(num <= f.max)) { visible = false; break; }
        }
      }
      tr.style.display = visible ? '' : 'none';
    }
  }

  inputs.forEach(el => {
    el.addEventListener('input', apply);
    el.addEventListener('change', apply);
    // Don't trigger sort when interacting with filter inputs.
    el.addEventListener('click', e => e.stopPropagation());
  });

  // Click-to-sort (only on the first header row).
  let sortState = { col: -1, dir: 1 };
  tbl.tHead.querySelectorAll('tr:first-child th').forEach((th, idx) => {
    th.addEventListener('click', () => {
      const type = th.dataset.sort || 'str';
      const dir = (sortState.col === idx) ? -sortState.dir : 1;
      sortState = { col: idx, dir };
      const rows = Array.from(tbody.rows);
      rows.sort((a, b) => {
        const va = a.cells[idx].dataset.v ?? a.cells[idx].textContent;
        const vb = b.cells[idx].dataset.v ?? b.cells[idx].textContent;
        if (type === 'num') return (parseFloat(va) - parseFloat(vb)) * dir;
        return va.localeCompare(vb) * dir;
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
})();
</script>
</body>
</html>
"""


def _kv_html(items) -> str:
    parts = []
    for k, v in items:
        parts.append(
            f'<div class="k">{_html.escape(str(k))}</div>'
            f'<div class="v">{_html.escape(str(v)) if v != "" else "&mdash;"}</div>'
        )
    return "".join(parts)


def _status_stats_html(status_counts: Counter, total: int) -> str:
    order = ["passed", "failed", "skipped", "crashed"]
    seen = set(status_counts) | set(order)
    parts = []
    for s in order + [x for x in seen if x not in order]:
        n = status_counts.get(s, 0)
        parts.append(
            f'<div class="stat"><div class="label">'
            f'<span class="pill {_html.escape(s)}">{_html.escape(s)}</span></div>'
            f'<div class="value">{n}</div></div>'
        )
    return "".join(parts)


def _bucket_bars_html(bucket_counts: Counter) -> str:
    labels = [f"{prev:>3d}-{ub:<3d}%" for prev, ub in zip([0] + _MEM_BUCKETS[:-1], _MEM_BUCKETS)]
    labels.append("100+ (OOM)")
    total = sum(bucket_counts.values()) or 1
    out = []
    for lab in labels:
        n = bucket_counts.get(lab, 0)
        pct = (n / total) * 100.0
        cls = "bar oom" if lab.startswith("100+") else "bar"
        out.append(
            f'<div class="bar-row">'
            f'<div>{_html.escape(lab.strip())}</div>'
            f'<div class="{cls}"><span style="width:{pct:.2f}%"></span></div>'
            f'<div style="text-align:right">{n}</div>'
            f'</div>'
        )
    return "".join(out)


def _rows_html(results: dict[str, dict]) -> str:
    rows = [_row_for(nid, rec) for nid, rec in results.items()]
    rows.sort(key=lambda r: (-r["peak_alloc_pct"], r["nodeid"]))
    out = []
    for r in rows:
        st = r["status"]
        pct = r["peak_alloc_pct"]
        bar_w = max(0.0, min(100.0, pct))
        bar_cls = "bar oom" if (r["oom"] or pct > 100) else "bar"
        out.append(
            f'<tr data-status="{_html.escape(st)}">'
            f'<td class="nodeid">{_html.escape(r["nodeid"])}</td>'
            f'<td><span class="pill {_html.escape(st)}">{_html.escape(st)}</span></td>'
            f'<td class="num" data-v="{r["duration_s"]}">{r["duration_s"]:.3f}</td>'
            f'<td class="num" data-v="{r["peak_alloc_GiB"]}">{r["peak_alloc_GiB"]:.3f}</td>'
            f'<td data-v="{pct}">'
            f'<div class="mem-cell"><div class="{bar_cls}"><span style="width:{bar_w:.2f}%"></span></div>'
            f'<span class="num">{pct:.2f}%</span></div></td>'
            f'<td>{"yes" if r["oom"] else ""}</td>'
            f'</tr>'
        )
    return "".join(out)


def _write_html_report(path: Path, *, results, status_counts, bucket_counts,
                       total_time, peak_max_pct, peak_max_node, crashed,
                       dev_info, sw) -> None:
    import datetime as _dt

    # Device KV
    if dev_info and dev_info.get("devices"):
        d_items = []
        for d in dev_info["devices"]:
            d_items.append((f'xpu[{d.get("index", "?")}]', d.get("name", "?")))
            d_items.append(("memory", _format_bytes_g(d.get("total_memory_B", 0))))
        device_kv = _kv_html(d_items)
    else:
        device_kv = _kv_html([("xpu", "not available")])

    env_order = ["os", "platform", "driver", "gcc", "python", "torch", "torchvision", "triton"]
    env_kv = _kv_html([(k, sw.get(k, "")) for k in env_order])

    n_tests = sum(status_counts.values())
    html_doc = (
        _HTML_TEMPLATE
        .replace("{{generated_at}}", _html.escape(_dt.datetime.now().isoformat(timespec="seconds")))
        .replace("{{n_tests}}", str(n_tests))
        .replace("{{total_time}}", f"{total_time:.2f}")
        .replace("{{device_kv}}", device_kv)
        .replace("{{env_kv}}", env_kv)
        .replace("{{status_stats}}", _status_stats_html(status_counts, n_tests))
        .replace("{{bucket_bars}}", _bucket_bars_html(bucket_counts))
        .replace("{{rows}}", _rows_html(results))
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")
