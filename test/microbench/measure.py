# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Measurement: protocol, device-time trace parsing, worker, validator, canary.

Runs inside worker processes started by ``run.py`` with each side's wheel.
"""

import bisect
import gc
import json
import math
import os
import platform
import socket
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace

from . import CANARY_ID
from .case import (
    alternatives,
    base_name,
    Case,
    case_log_dir,
    case_log_name,
    dtype_name,
    expected_names,
    functional_name,
    load_cases,
    OpBench,
    State,
    torch,
)

# ---------------------------------------------------------------------------
# Device time from the profiler trace (no torch needed)
# ---------------------------------------------------------------------------
# Every measured iteration runs inside a ``__mb_iter__`` host annotation.
# Device events (kernel, memcpy, memset) are attributed to an iteration
# through the ``External id`` of the host op that launched them, falling back
# to the runtime launch event linked by ``correlation``; events launched
# outside the windows (warm-up pass, input reset) are not counted. An
# iteration is invalid when it has no device event or when a kernel launch
# inside its window has no device event (incomplete trace).
# ``key_averages().self_device_time_total`` is not used: it is unreliable on
# XPU.

ITER_TAG = "__mb_iter__"
DEVICE_CATS = {"kernel", "gpu_kernel", "xpu_kernel", "gpu_memcpy", "gpu_memset"}
HOST_OP_CATS = {"cpu_op", "user_annotation", "python_function"}


def _is_runtime(cat):
    return "runtime" in cat or cat == "xpu_driver"


def _is_kernel_launch(name):
    name = (name or "").lower()
    return "launch" in name and "kernel" in name


def _span(e):
    return float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))


def _annotations(events, name):
    return sorted(
        _span(e)
        for e in events
        if e.get("ph") == "X"
        and e.get("cat") == "user_annotation"
        and e.get("name") == name
    )


class _Windows:
    """Find the (start, end, tag) window containing a timestamp."""

    def __init__(self, windows):
        self.windows = sorted(windows, key=lambda w: w[0])
        self.starts = [w[0] for w in self.windows]

    def find(self, ts):
        i = bisect.bisect_right(self.starts, ts) - 1
        if i >= 0 and ts <= self.windows[i][1]:
            return self.windows[i][2]
        return None


def _device_events(events):
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS:
            yield e, e.get("args") or {}


def attribute_device_time(events, detail=False):
    """Per-iteration device time (us; ``None`` = invalid) plus diagnostics.

    ``detail`` adds ``iter_events``: the device events counted per iteration
    as ``(name, category, dur_us)`` tuples, for the per-case device log.
    """
    iter_w = _annotations(events, ITER_TAG)
    windows = _Windows([(s, t, i) for i, (s, t) in enumerate(iter_w)])
    ext_window, corr_window, launches = {}, {}, {}
    for e in events:
        cat = e.get("cat", "")
        if e.get("ph") != "X" or not (cat in HOST_OP_CATS or _is_runtime(cat)):
            continue
        w = windows.find(float(e["ts"]))
        if w is None:
            continue
        args = e.get("args") or {}
        if args.get("External id") is not None and cat != "user_annotation":
            ext_window.setdefault(args["External id"], w)
        if args.get("correlation") is not None and _is_runtime(cat):
            corr_window[args["correlation"]] = w
            if _is_kernel_launch(e.get("name")):
                launches[args["correlation"]] = w

    n = len(iter_w)
    samples, iter_kernels = [0.0] * n, [0] * n
    iter_events = [[] for _ in range(n)]
    seen, excluded = set(), 0
    for e, args in _device_events(events):
        w = ext_window.get(args.get("External id"))
        if w is None:
            w = corr_window.get(args.get("correlation"))
        if w is None:
            excluded += 1
            continue
        seen.add(args.get("correlation"))
        samples[w] += float(e.get("dur", 0))
        iter_kernels[w] += 1
        iter_events[w].append(
            (e.get("name"), e.get("cat"), round(float(e.get("dur", 0)), 3))
        )
    missing = [0] * n
    for corr, w in launches.items():
        if corr not in seen:
            missing[w] += 1
    out = [samples[i] if iter_kernels[i] and not missing[i] else None for i in range(n)]
    res = {
        "samples": out,
        "iterations": n,
        "invalid": out.count(None),
        "excluded_device_events": excluded,
        "iter_kernels": iter_kernels,
        "missing_launches": missing,
    }
    if detail:
        res["iter_events"] = iter_events
    return res


def host_ops_in_iterations(events):
    """``cpu_op`` events inside ``__mb_iter__`` windows, for the validator.

    Each is ``{name, ts, end, ext, tid, kernels, args}``; ``kernels`` counts
    device events launched directly by the op.
    """
    windows = _Windows(
        [(s, t, i) for i, (s, t) in enumerate(_annotations(events, ITER_TAG))]
    )
    ops, by_ext, corr_to_ext = [], {}, {}
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, args = e.get("cat", ""), e.get("args") or {}
        if cat == "cpu_op" and windows.find(float(e["ts"])) is not None:
            ts, end = _span(e)
            ext = args.get("External id")
            op = {"name": e.get("name"), "ts": ts, "end": end, "ext": ext}
            op.update(tid=e.get("tid"), kernels=0, args=args)
            ops.append(op)
            if ext is not None:
                by_ext.setdefault(ext, op)
        elif _is_runtime(cat) and args.get("correlation") is not None:
            corr_to_ext[args["correlation"]] = args.get("External id")
    for _, args in _device_events(events):
        ext = args.get("External id")
        op = by_ext.get(
            corr_to_ext.get(args.get("correlation")) if ext is None else ext
        )
        if op is not None:
            op["kernels"] += 1
    return ops


def subtree_kernels(ops):
    """Annotate each host op with kernels launched by it or any nested op."""
    by_tid = {}
    for op in ops:
        by_tid.setdefault(op["tid"], []).append(op)
    for lst in by_tid.values():
        lst.sort(key=lambda o: (o["ts"], -o["end"]))
        stack = []
        for op in lst:
            op["subtree_kernels"] = op["kernels"]
            while stack and op["ts"] >= stack[-1]["end"]:
                stack.pop()
            op["parent"] = stack[-1]["name"] if stack else None
            for anc in stack:
                anc["subtree_kernels"] += op["kernels"]
            stack.append(op)
    return ops


# ---------------------------------------------------------------------------
# Measurement protocol (continuous)
# ---------------------------------------------------------------------------
# The op runs back to back, as in a model, with no cache flush and no sync
# between iterations, in passes of ``ring`` iterations with a sync after each
# pass. A measured pass follows at least one warm-up pass, so it pays for
# writing back the outputs the previous pass left dirty in L2, roughly as
# much as it leaves dirty itself: the result is the steady-state time.
#
# * Inputs: ``ring`` distinct input sets, one per iteration of a pass, so
#   that a pass touches at least twice the L2 size (``ring >= 2 * (L2 /
#   (input + output) + 1)``, at least ``min_ring``). Inputs above
#   ``reuse_input_bytes`` cannot stay in L2 and use one set.
# * Outputs: up to ``keep_output_bytes`` the output of every iteration stays
#   alive for ``ring`` iterations, so each iteration writes to a new address
#   instead of rewriting the L2 lines of the previous output. The warm-up pass
#   leaves the allocator with enough cached blocks that no device allocation
#   happens while measuring.
# * In-place cases reset all input sets between passes (outside the
#   iteration windows).
#
# The E2E pass times whole passes with ``time.perf_counter`` (one sample per
# pass: pass time / ring) without the profiler; the device pass repeats the
# warm-up and measured passes under the profiler and gives one sample per
# iteration. Warm-up is time based until consecutive passes agree; then at
# least ``min_iters`` iterations and ``min_total_ms`` of op time.


@dataclass
class ProtocolConfig:
    warmup_ms: float = 200.0
    warmup_max_ms: float = 3000.0
    warmup_stable_passes: int = 3
    warmup_stable_tol: float = 0.05
    min_iters: int = 50
    max_iters: int = 400
    min_total_ms: float = 40.0
    min_passes: int = 2
    l2_bytes: int = 32 * 1024 * 1024
    min_ring: int = 10
    max_ring: int = 256
    reuse_input_bytes: int = 100 * 1024 * 1024
    keep_output_bytes: int = 100 * 1024 * 1024
    max_mem_frac: float = 0.5
    seed: int = 1234
    device_pass: bool = True
    max_invalid_frac: float = 0.1

    @classmethod
    def load(cls, path=None, quick=False):
        """Fields from a JSON file; ``quick``: 3 iterations, no warmup (smoke)."""
        d = json.load(open(path)) if path else {}
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if quick:
            cfg.min_iters = cfg.max_iters = 3
            cfg.min_ring = cfg.max_ring = 3
            cfg.min_passes = 1
            cfg.warmup_ms = cfg.warmup_max_ms = 0
            cfg.warmup_stable_passes = 1
        return cfg


class Device:
    """Sync and memory info of ``xpu`` (or ``cpu`` for smoke tests)."""

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg

    def sync(self):
        if self.name == "xpu":
            torch.xpu.synchronize()

    def free_bytes(self):
        return torch.xpu.mem_get_info()[0] if self.name == "xpu" else 1 << 40

    def allocated(self):
        return torch.xpu.memory_allocated() if self.name == "xpu" else 0

    def empty_cache(self):
        if self.name == "xpu":
            torch.xpu.empty_cache()

    def profiler_activities(self):
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self.name == "xpu":
            acts.append(torch.profiler.ProfilerActivity.XPU)
        return acts


def state_bytes(obj, seen=None):
    """Bytes held by tensors reachable from a setup() state or an output."""
    seen = set() if seen is None else seen
    if isinstance(obj, torch.Tensor):
        storage = obj.untyped_storage()
        if storage.data_ptr() in seen:
            return 0
        seen.add(storage.data_ptr())
        return storage.nbytes()
    if isinstance(obj, SimpleNamespace):
        obj = vars(obj)
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(state_bytes(v, seen) for v in obj)
    return 0


class Workload:
    """Input sets and the output ring of one case in this process."""

    def __init__(self, case, dev):
        cfg = dev.cfg
        self.case, self.dev = case, dev
        torch.manual_seed(cfg.seed)
        before = dev.allocated()
        first = case.build(dev.name)
        dev.sync()
        # backward graphs hold tensors the state does not reference
        self.set_bytes = max(1, state_bytes(first), dev.allocated() - before)
        out = case.run(first)
        dev.sync()
        self.out_bytes = state_bytes(out)
        del out
        case.reset(first)
        self.keep_outputs = self.out_bytes <= cfg.keep_output_bytes
        traffic = state_bytes(first) + self.out_bytes
        ring = 2 * (cfg.l2_bytes // max(1, traffic) + 1)
        ring = min(cfg.max_ring, max(cfg.min_ring, ring))
        reuse = self.set_bytes > cfg.reuse_input_bytes
        per_iter = (0 if reuse else self.set_bytes) + (
            self.out_bytes if self.keep_outputs else 0
        )
        budget = int(dev.free_bytes() * cfg.max_mem_frac)
        if per_iter and ring * per_iter > budget:
            ring = max(2, budget // per_iter)
        self.ring = ring
        self.sets = [first]
        for i in range(1, 1 if reuse else ring):
            torch.manual_seed(cfg.seed + i)
            self.sets.append(case.build(dev.name))
        self.outputs = [None] * ring
        dev.sync()

    def info(self):
        return {
            "input_sets": len(self.sets),
            "set_bytes": self.set_bytes,
            "output_bytes": self.out_bytes,
            "ring": self.ring,
            "keep_outputs": self.keep_outputs,
        }

    def reset(self):
        if self.case.inplace:
            for s in self.sets:
                self.case.reset(s)
            self.dev.sync()

    def run_pass(self, tag=None):
        """One pass of ``ring`` iterations, back to back, then a sync."""
        from torch.profiler import record_function

        sets, outputs, case = self.sets, self.outputs, self.case
        for i in range(self.ring):
            s = sets[i % len(sets)]
            if tag:
                with record_function(tag):
                    out = case.run(s)
            else:
                out = case.run(s)
            outputs[i] = out if self.keep_outputs else None
            del out
        self.dev.sync()

    def timed_pass(self):
        """E2E of one pass, us per iteration."""
        self.reset()
        t0 = time.perf_counter()
        self.run_pass()
        return (time.perf_counter() - t0) * 1e6 / self.ring

    def release(self):
        self.sets, self.outputs = [], []


def warmup(w):
    """Passes until ``warmup_ms`` and consecutive passes agree; returns
    (us per iteration, info)."""
    cfg = w.dev.cfg
    start = time.perf_counter()
    times, stable = [], False
    while True:
        times.append(w.timed_pass())
        elapsed_ms = (time.perf_counter() - start) * 1e3
        last = times[-cfg.warmup_stable_passes :]
        if len(last) == cfg.warmup_stable_passes:
            lo, hi = min(last), max(last)
            stable = lo > 0 and (hi - lo) / lo <= cfg.warmup_stable_tol
        if (elapsed_ms >= cfg.warmup_ms and stable) or elapsed_ms >= cfg.warmup_max_ms:
            return times[-1], {
                "passes": len(times),
                "ms": round(elapsed_ms, 1),
                "stable": stable,
                "pass_medians_us": [round(t, 3) for t in times],
            }


def decide_passes(cfg, ring, per_iter_us):
    need = math.ceil(cfg.min_total_ms * 1e3 / max(per_iter_us, 1e-3))
    iters = min(cfg.max_iters, max(cfg.min_iters, need))
    return max(cfg.min_passes, math.ceil(iters / ring))


def profile_passes(w, passes, keep_trace=None):
    """One warm-up pass and ``passes`` tagged passes under the profiler;
    returns the trace events."""
    from torch.profiler import profile

    with profile(activities=w.dev.profiler_activities()) as prof:
        w.reset()
        w.run_pass()
        for _ in range(passes):
            w.reset()
            w.run_pass(ITER_TAG)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="mb_trace_")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f).get("traceEvents", [])
        if keep_trace:
            os.replace(path, keep_trace)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return events


def measure(case, dev, keep_trace=None, detail=False):
    """Full measurement of one case in this process. Returns a result dict.

    ``detail`` adds ``_device_detail`` (per-iteration device events, see
    ``attribute_device_time``) for the per-case log; the worker strips it.
    """
    w = Workload(case, dev)
    est, wu = warmup(w)
    passes = decide_passes(dev.cfg, w.ring, est)
    rec = dict(w.info(), warmup=wu, passes=passes, iters=passes * w.ring)
    rec["e2e_us"] = [round(w.timed_pass(), 3) for _ in range(passes)]
    rec["device_us"], rec["device_status"] = None, "unavailable"
    if dev.cfg.device_pass and dev.name != "cpu":
        att = attribute_device_time(profile_passes(w, passes, keep_trace), detail)
        rec["device_us"] = [None if x is None else round(x, 3) for x in att["samples"]]
        rec["device_invalid"] = att["invalid"]
        rec["device_excluded"] = att["excluded_device_events"]
        ok = (
            att["iterations"]
            and att["invalid"] <= dev.cfg.max_invalid_frac * att["iterations"]
        )
        rec["device_status"] = "ok" if ok else "invalid_attribution"
        if detail:
            rec["_device_detail"] = att
    w.release()
    del w
    gc.collect()
    dev.empty_cache()
    return rec


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------
# A fixed kernel run on every card at the start and end of a run. A canary
# that deviates from the card's history (or drifts between start and end)
# makes the run's verdicts inconclusive.


class CanaryBench(OpBench):
    op = "__canary__"
    expected = {"fwd": ("add",)}
    params = [dict(shape=(64 * 1024 * 1024,), dtype="fp32")]
    gating = False

    def setup(self, p, device):
        x = torch.ones(p["shape"], device=device)
        return State(x=x, y=torch.ones_like(x))

    def fwd(self, s):
        return torch.add(s.x, s.y)


def canary_case():
    case = Case(CanaryBench(), dict(CanaryBench.params[0]), "fwd", module="canary")
    case.case_id = CANARY_ID
    return case


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------
# The scheduler starts one worker per (card, side, round) with the venv python
# of that side. Output is JSONL: one ``env`` record, then a ``start`` record
# (crash locator) and a ``result`` record per case.


def _emit(fh, obj):
    fh.write(json.dumps(obj) + "\n")
    fh.flush()


def env_info(device):
    import torch

    info = {
        "type": "env",
        "time": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "device": device,
        "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK"),
        "ze_tracing_layer": os.environ.get("ZE_ENABLE_TRACING_LAYER"),
    }
    if device == "xpu" and torch.xpu.is_available():
        props = torch.xpu.get_device_properties()
        info["device_name"] = props.name
        info["driver_version"] = getattr(props, "driver_version", None)
        info["total_memory"] = props.total_memory
    return info


def process_log_name(case, side, stage, rnd, kind, ext="log"):
    """``<op>_<pass>_<stage>[_<side>]_proc<k>_<kind>.log`` of one worker process."""
    parts = [stage] if side == stage else [stage, side]
    name = "_".join([*parts, f"proc{rnd}_{kind}"])
    if case.case_id != CANARY_ID:
        name = case_log_name(case.op, case.pass_, name)
    return f"{name}.{ext}"


def write_process_logs(log_dir, case, tag, rec, env):
    """E2E and device-time raw logs of one case measured in this process."""
    os.makedirs(log_dir, exist_ok=True)
    head = [
        f"case {case.case_id} {case.op}/{case.pass_} {case.label}",
        f"side {tag['side']} stage {tag['stage']} process {tag['round']} card {tag['card']} "
        f"pid {os.getpid()} torch {env.get('torch_git_version')}",
    ]
    wu = rec.get("warmup") or {}
    lines = head + [
        f"continuous passes of {rec.get('ring')} iterations; input sets: {rec.get('input_sets')} "
        f"({rec.get('set_bytes')} bytes each, set i % {rec.get('input_sets')} in iteration i); "
        f"output {rec.get('output_bytes')} bytes, kept alive for a pass: {rec.get('keep_outputs')}",
        f"warmup: {wu.get('passes')} passes in {wu.get('ms')} ms, stable={wu.get('stable')}",
        "warmup pass times (us per iteration): "
        + " ".join(str(m) for m in wu.get("pass_medians_us") or []),
        f"measured: {rec.get('passes')} passes ({rec.get('iters')} iterations)",
        "pass e2e_us_per_iteration",
    ]
    lines += [f"{i} {v}" for i, v in enumerate(rec.get("e2e_us") or [])]
    side, stage, rnd = tag["side"], tag["stage"], tag["round"]
    with open(
        os.path.join(log_dir, process_log_name(case, side, stage, rnd, "e2e")), "w"
    ) as f:
        f.write("\n".join(lines) + "\n")
    det = rec.get("_device_detail")
    lines = head + [f"device status: {rec.get('device_status')}"]
    if det is None:
        lines.append("no device pass (device time unavailable on this device)")
    else:
        lines.append(
            f"iterations {det['iterations']}, invalid {det['invalid']}, device events outside "
            f"the iterations (warm-up pass, input reset) {det['excluded_device_events']}"
        )
        for i, sample in enumerate(det["samples"]):
            if sample is None:
                why = (
                    f"{det['missing_launches'][i]} kernel launch(es) without a device event"
                    if det["missing_launches"][i]
                    else "no device event attributed to the iteration"
                )
                lines.append(f"iter {i}: INVALID ({why})")
            else:
                lines.append(f"iter {i}: {sample:.3f} us")
            lines += [
                f"  counted {cat} {name} {dur}"
                for name, cat, dur in det["iter_events"][i]
            ]
    with open(
        os.path.join(log_dir, process_log_name(case, side, stage, rnd, "device")), "w"
    ) as f:
        f.write("\n".join(lines) + "\n")


def run_worker(args):
    # The tracing layer must be on before the Level Zero driver initializes.
    if args.device == "xpu":
        os.environ.setdefault("ZE_ENABLE_TRACING_LAYER", "1")
    with open(args.cases) as f:
        wanted = json.load(f)
    cfg = ProtocolConfig.load(args.config, args.quick)
    cases = load_cases()
    cases[CANARY_ID] = canary_case()
    dev = Device(args.device, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tag = {
        "side": args.side,
        "round": args.round,
        "card": args.card,
        "stage": args.stage,
    }
    env = env_info(args.device)
    with open(args.out, "a") as fh:
        _emit(fh, dict(env, **tag))
        for cid in wanted:
            case = cases.get(cid)
            base = dict(tag, case_id=cid, pid=os.getpid())
            if case is None:
                _emit(fh, dict(base, type="result", status="unknown_case"))
                continue
            _emit(fh, dict(base, type="start", time=time.time()))
            t0 = time.time()
            case_logs = (
                os.path.join(args.log_dir, case_log_dir(case.op, case.pass_, cid))
                if args.log_dir
                else None
            )
            trace = None
            if case_logs and args.keep_traces:
                os.makedirs(case_logs, exist_ok=True)
                trace = os.path.join(
                    case_logs,
                    process_log_name(
                        case, args.side, args.stage, args.round, "trace", "json"
                    ),
                )
            try:
                rec, status = (
                    measure(case, dev, keep_trace=trace, detail=bool(case_logs)),
                    "ok",
                )
                if case_logs:
                    write_process_logs(case_logs, case, tag, rec, env)
            except Exception as e:  # noqa: BLE001 - record and continue
                rec = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=8),
                }
                status = "error"
                dev.empty_cache()
            rec.pop("_device_detail", None)
            rec.update(base, type="result", status=status, op=case.op, pass_=case.pass_)
            rec["seconds"] = round(time.time() - t0, 3)
            _emit(fh, rec)
    return 0


def add_worker_parser(sub):
    p = sub.add_parser("worker", help="(internal) measure cases in this process")
    p.add_argument("--cases", required=True, help="JSON list of case_ids")
    p.add_argument("--out", required=True, help="JSONL output (appended)")
    p.add_argument("--config", default=None, help="protocol config JSON")
    p.add_argument("--device", default="xpu")
    p.add_argument("--side", default="B")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--card", default="0")
    p.add_argument("--stage", default="screen")
    p.add_argument("--quick", action="store_true", help="3 iterations, no warmup")
    p.add_argument(
        "--log-dir", default=None, help="write per-case raw logs under <dir>/<case_id>/"
    )
    p.add_argument(
        "--keep-traces",
        action="store_true",
        help="also save the device-pass chrome trace (debug)",
    )
    p.set_defaults(func=lambda a: sys.exit(run_worker(a)))


# ---------------------------------------------------------------------------
# Case validator
# ---------------------------------------------------------------------------
# Runs each case once, untimed, and checks that it measures what it claims:
#
# 1. every ``expected_aten_ops`` entry is dispatched inside ``run()`` and, on a
#    GPU, launches device work (not a CPU fallback);
# 2. the first expected op receives the declared dtype/shape/stride;
# 3. ``run()`` does not build inputs (random factories, host/device copies)
#    unless the op under test is such an op;
# 4. every kernel launched by ``run()`` shows up in the trace and is
#    attributed to its iteration.
#
# The dispatched-op log doubles as the coverage source.

CONSTRUCTION_OPS = {
    "randn",
    "rand",
    "randint",
    "randn_like",
    "rand_like",
    "randint_like",
    "normal",
    "normal_",
    "uniform_",
    "random_",
    "randperm",
    "bernoulli",
    "bernoulli_",
}


def _tensor_info(t):
    stride = tuple(t.stride()) if t.layout == torch.strided else None
    return {
        "shape": tuple(t.shape),
        "dtype": dtype_name(t.dtype),
        "stride": stride,
        "device": t.device.type,
    }


def dispatch_log(fn):
    """Run ``fn()``; return the aten ops it dispatched with their tensor args."""
    from torch.utils._python_dispatch import TorchDispatchMode

    log = []

    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            items = []
            for a in list(args) + list(kwargs.values()):
                items += a if isinstance(a, (list, tuple)) else [a]
            tensors = [_tensor_info(t) for t in items if isinstance(t, torch.Tensor)]
            log.append({"op": str(func._schema.name), "tensors": tensors})
            return func(*args, **kwargs)

    with Mode():
        fn()
    return log


def _match_inputs(expected, actual):
    errs = []
    for i, spec in enumerate(expected):
        if spec is None:
            continue
        if i >= len(actual):
            errs.append(f"input {i}: missing (expected {spec})")
            continue
        for key in ("shape", "dtype", "stride"):
            if spec.get(key) is None:
                continue
            want = spec[key] if key == "dtype" else tuple(spec[key])
            if actual[i][key] != want:
                errs.append(
                    f"input {i} {key}: declared {want}, actual {actual[i][key]}"
                )
    return errs


def profile_runs(case, dev, s, n):
    """``n`` tagged runs of ``s`` under the profiler; returns the trace events."""
    from torch.profiler import profile, record_function

    with profile(activities=dev.profiler_activities()) as prof:
        for _ in range(n):
            case.reset(s)
            dev.sync()
            with record_function(ITER_TAG):
                out = case.run(s)
            dev.sync()
            del out
    fd, path = tempfile.mkstemp(suffix=".json", prefix="mb_trace_")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            return json.load(f).get("traceEvents", [])
    finally:
        os.unlink(path)


def validate_case(case, dev):
    """Return a result dict with ``status`` ('pass'/'fail') and ``failures``."""
    device = dev.name
    torch.manual_seed(dev.cfg.seed)
    s = case.build(device)
    # warm run so lazy init is not attributed to the case
    case.reset(s)
    case.run(s)
    dev.sync()
    case.reset(s)
    dev.sync()
    log = dispatch_log(lambda: case.run(s))
    dev.sync()

    dispatched = [e["op"] for e in log]
    dispatched_funcs = {functional_name(o) for o in dispatched}
    expected = list(case.expected_aten_ops)
    # each entry may list alternatives 'a|b'; resolve to the one dispatched
    resolved = {
        e: next((a for a in alternatives(e) if a in dispatched_funcs), None)
        for e in expected
    }
    expected_funcs = {f for f in resolved.values() if f} | expected_names(expected)
    observed = ", ".join(dict.fromkeys(base_name(o) for o in dispatched)) or "<none>"
    failures = [
        f"expected aten op '{e}' not executed on {device.upper()}; observed: {observed}"
        for e, hit in resolved.items()
        if hit is None
    ]

    spec = case.bench.expect_inputs(case.params, case.pass_)
    first = resolved[expected[0]] if expected else None
    if spec and first:
        e = next(e for e in log if functional_name(e["op"]) == first)
        failures += [
            f"'{base_name(e['op'])}' {err}" for err in _match_inputs(spec, e["tensors"])
        ]

    for e in log:
        name, fname = base_name(e["op"]), functional_name(e["op"])
        devs = {t["device"] for t in e["tensors"]}
        if fname in expected_funcs:
            cpu_args = [
                t["shape"]
                for t in e["tensors"]
                if t["device"] == "cpu" and t["shape"] != ()
            ]
            if device != "cpu" and cpu_args:
                failures.append(f"'{name}' received CPU tensor inputs {cpu_args}")
        elif name in CONSTRUCTION_OPS:
            failures.append(f"timed region constructs inputs: aten::{name}")
        elif (
            name in ("_to_copy", "copy_")
            and device != "cpu"
            and "cpu" in devs
            and len(devs) > 1
        ):
            failures.append(f"timed region contains a host/device copy: aten::{name}")

    kernel_ops = attribution = None
    if device != "cpu":
        events = profile_runs(case, dev, s, 2)
        ops = subtree_kernels(host_ops_in_iterations(events))
        kernel_ops = sorted({base_name(o["name"]) for o in ops if o["kernels"] > 0})
        for fname in filter(None, resolved.values()):
            inst = [o for o in ops if functional_name(o["name"]) == fname]
            if inst and not any(o["subtree_kernels"] > 0 for o in inst):
                failures.append(
                    f"expected aten op '{fname}' launched no {device.upper()} kernel (CPU fallback?)"
                )
        att = attribute_device_time(events)
        attribution = {
            k: att[k] for k in ("invalid", "iter_kernels", "missing_launches")
        }
        if att["invalid"]:
            failures.append(
                f"device time not attributable in {att['invalid']}/{att['iterations']} iterations "
                f"(op kernels: {att['iter_kernels']}, launches without a device event: "
                f"{att['missing_launches']})"
            )

    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "dispatched_ops": list(dict.fromkeys(base_name(o) for o in dispatched)),
        "kernel_ops": kernel_ops,
        "attribution": attribution,
    }


def run_validate(args):
    if args.device == "xpu":
        os.environ.setdefault("ZE_ENABLE_TRACING_LAYER", "1")
    cases = load_cases()
    if args.cases:
        with open(args.cases) as f:
            ids = json.load(f)
    else:
        ops = args.op.split(",") if args.op else None
        ids = [c.case_id for c in cases.values() if not ops or c.op in ops]
    dev = Device(args.device, ProtocolConfig.load(args.config))
    deadline = time.time() + args.budget_min * 60 if args.budget_min else None
    n_fail = n_skip = 0
    with open(args.out, "w") if args.out else sys.stdout as fh:
        for cid in ids:
            case = cases.get(cid)
            if case is not None and deadline and time.time() > deadline:
                res = {
                    "status": "not_validated",
                    "failures": ["validation time budget exhausted"],
                }
                res.update(case_id=cid, op=case.op, pass_=case.pass_, label=case.label)
                n_skip += 1
                fh.write(json.dumps(res) + "\n")
                continue
            if case is None:
                res = {
                    "status": "fail",
                    "failures": [f"unknown case_id {cid}"],
                    "case_id": cid,
                }
            else:
                try:
                    res = validate_case(case, dev)
                except Exception as e:  # noqa: BLE001 - a crashing case is a failing case
                    res = {
                        "status": "fail",
                        "failures": [f"case raised {type(e).__name__}: {e}"],
                        "traceback": traceback.format_exc(limit=6),
                    }
                # return the case's cached blocks: a later large case must not
                # run out of memory because earlier cases left the cache full
                gc.collect()
                dev.empty_cache()
                res.update(case_id=cid, op=case.op, pass_=case.pass_, label=case.label)
            n_fail += res["status"] == "fail"
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if args.verbose or res["status"] == "fail":
                head = f"[{res['status'].upper()}] {res.get('op')}/{res.get('pass_')} {res.get('label', cid)}"
                print(head, file=sys.stderr)
                for f in res["failures"]:
                    print(f"    - {f}", file=sys.stderr)
    print(
        f"validated {len(ids) - n_skip} case(s), {n_fail} failed, {n_skip} skipped (time budget)",
        file=sys.stderr,
    )
    return 1 if n_fail else 0


def add_validate_parser(sub):
    p = sub.add_parser("validate", help="check that cases measure the op they declare")
    p.add_argument("--cases", default=None, help="JSON list of case_ids (default: all)")
    p.add_argument("--op", default=None, help="comma separated op names")
    p.add_argument("--out", default=None, help="JSONL output (default stdout)")
    p.add_argument("--device", default="xpu")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--budget-min",
        type=float,
        default=0,
        help="skip the remaining cases after this (0: no limit)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=lambda a: sys.exit(run_validate(a)))
