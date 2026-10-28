# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Scheduler and machine environment (no torch in this process).

``python -m microbench schedule`` drives worker processes over the cards
(A/B, A/A, single-sided or multi-wheel); ``python -m microbench env`` shows
the layout, locks GPU frequency and runs the pre-run checks.
"""

import copy
import csv
import glob
import hashlib
import json
import os
import platform
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

from . import (
    CONFIG_DIR,
    DEFAULT_CASE_SECONDS,
    judge,
    load_history,
    load_json,
    load_yaml,
    PACKAGE_DIR,
    PROTOCOL_VERSION,
    read_jsonl,
    reference,
)
from .case import case_log_dir, case_log_name, load_cases, manifest as build_manifest
from .selection import candidate_commits, Config

# ---------------------------------------------------------------------------
# Machine layout, GPU frequency, pre-run checks
# ---------------------------------------------------------------------------
# Cards, NUMA nodes and CPU sets come from ``machines`` in ``config/gating.yaml`` when the
# host has an entry there, otherwise from sysfs. Frequency control supports
# xe (BMG: ``device/tile*/gt*/freq0/*_freq``) and i915 (``gt_*_freq_mhz``).
# ``lock`` pins min=max at ``ratio * RP0`` (default 0.95), slightly below the
# maximum so power/thermal throttling does not kick in mid run.

SYSFS = "/sys/class/drm"
FREQ_FILES = {
    "xe": {
        "min": "min_freq",
        "max": "max_freq",
        "cur": "cur_freq",
        "act": "act_freq",
        "rp0": "rp0_freq",
        "rpn": "rpn_freq",
    },
    "i915": {
        "min": "gt_min_freq_mhz",
        "max": "gt_max_freq_mhz",
        "cur": "gt_cur_freq_mhz",
        "act": "gt_act_freq_mhz",
        "rp0": "gt_RP0_freq_mhz",
        "rpn": "gt_RPn_freq_mhz",
    },
}


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _write(path, value):
    with open(path, "w") as f:
        f.write(str(value))


def parse_cpulist(text):
    cpus = []
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        a, _, b = part.partition("-")
        cpus.extend(range(int(a), int(b or a) + 1))
    return cpus


def format_cpulist(cpus):
    out = []
    for c in sorted(cpus):
        if out and out[-1][1] == c - 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def detect_cards():
    """Intel display-class DRM cards, ordered by PCI address (= ZE index)."""
    cards = []
    for path in glob.glob(os.path.join(SYSFS, "card[0-9]*")):
        dev = os.path.join(path, "device")
        if not re.fullmatch(r"card\d+", os.path.basename(path)):
            continue
        if _read(f"{dev}/vendor") != "0x8086" or not _read(
            f"{dev}/class", ""
        ).startswith("0x03"):
            continue
        cards.append(
            {
                "drm": os.path.basename(path),
                "bdf": os.path.basename(os.path.realpath(dev)),
                "numa": max(int(_read(f"{dev}/numa_node", "-1") or -1), 0),
                "local_cpus": parse_cpulist(_read(f"{dev}/local_cpulist", "")),
            }
        )
    cards.sort(key=lambda c: c["bdf"])
    by_node = {}
    for i, c in enumerate(cards):
        c["index"] = i
        by_node.setdefault(c["numa"], []).append(c)
    # each card gets a disjoint CPU slice of its NUMA node
    for node, group in by_node.items():
        cpus = (
            group[0]["local_cpus"]
            or parse_cpulist(_read(f"/sys/devices/system/node/node{node}/cpulist", ""))
            or list(range(os.cpu_count() or 1))
        )
        per = max(1, len(cpus) // len(group))
        for k, c in enumerate(group):
            c["cpus"] = format_cpulist(cpus[k * per : (k + 1) * per])
    return cards


def machine_config(path=None):
    """Card layout and settings for this host; config entries win over sysfs."""
    host = socket.gethostname()
    data = load_yaml(path or os.path.join(CONFIG_DIR, "gating.yaml"), {})
    data = data.get("machines", data)
    cfg = dict(data.get("default", {}))
    for pattern, entry in (data.get("hosts") or {}).items():
        if re.fullmatch(pattern, host):
            cfg.update(entry)
            break
    if not cfg.get("cards"):
        cards = detect_cards()
        # a shared CI runner owns only the cards in its ZE_AFFINITY_MASK
        mask = os.environ.get("ZE_AFFINITY_MASK", "").strip()
        if mask:
            own = [int(m.split(".")[0]) for m in mask.split(",") if m.strip()]
            cards = [c for c in cards if c["index"] in own] or [
                {"index": i} for i in own
            ]
        cfg["cards"] = cards[: cfg.get("max_cards")]
    cfg["host"] = host
    return cfg


def bind_prefix(card, binding):
    """Command prefix pinning a worker to the card's NUMA node and CPUs."""
    cpus, node = card.get("cpus"), card.get("numa", 0)
    if binding == "numactl" and shutil.which("numactl"):
        cpu_opt = f"--physcpubind={cpus}" if cpus else f"--cpunodebind={node}"
        return ["numactl", f"--membind={node}", cpu_opt]
    if binding in ("numactl", "taskset") and cpus and shutil.which("taskset"):
        return ["taskset", "-c", cpus]
    return []


def read_freq(drm):
    """One dict per GT frequency domain of a card: {kind, dir, min, max, ...}."""
    base = os.path.join(SYSFS, drm)
    doms = [("xe", d) for d in sorted(glob.glob(f"{base}/device/tile*/gt*/freq0"))]
    if not doms and os.path.exists(f"{base}/gt_min_freq_mhz"):
        doms = [("i915", base)]
    out = []
    for kind, d in doms:
        entry = {"kind": kind, "dir": d}
        for k, name in FREQ_FILES[kind].items():
            v = _read(os.path.join(d, name))
            entry[k] = int(v) if v and v.isdigit() else None
        out.append(entry)
    return out


def read_temps(drm):
    temps = {}
    for f in glob.glob(os.path.join(SYSFS, drm, "device/hwmon/hwmon*/temp*_input")):
        v = _read(f)
        if v and v.lstrip("-").isdigit():
            temps[_read(f.replace("_input", "_label")) or os.path.basename(f)] = (
                int(v) / 1000
            )
    return temps


def lock_gpu(drm, ratio=0.95, mhz=None):
    done = []
    for dom in read_freq(drm):
        names = FREQ_FILES[dom["kind"]]
        target = mhz or int((dom["rp0"] or 0) * ratio // 50 * 50)
        if not target:
            raise RuntimeError(f"{drm}: cannot determine RP0 in {dom['dir']}")
        # raise max first when going up, lower min first when going down
        order = ("max", "min") if target > (dom["max"] or 0) else ("min", "max")
        for k in order:
            _write(os.path.join(dom["dir"], names[k]), target)
        done.append({"dir": dom["dir"], "target": target})
    return done


def unlock_gpu(drm):
    for dom in read_freq(drm):
        names = FREQ_FILES[dom["kind"]]
        for key, src in (("max", "rp0"), ("min", "rpn")):
            if dom[src]:
                _write(os.path.join(dom["dir"], names[key]), dom[src])


def card_busy_pids(drm):
    """PIDs (other than ours) holding the card's render or primary node open."""
    nodes = {f"/dev/dri/{drm}"} | {
        f"/dev/dri/{os.path.basename(r)}"
        for r in glob.glob(os.path.join(SYSFS, drm, "device/drm/renderD*"))
    }
    pids = set()
    for fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = int(fd_dir.split("/")[2])
        try:
            fds = os.listdir(fd_dir) if pid != os.getpid() else []
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(os.path.join(fd_dir, fd)) in nodes:
                    pids.add(pid)
                    break
            except OSError:
                continue
    return sorted(pids)


def cpu_governors():
    files = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor")
    return sorted({v for v in map(_read, files) if v})


def check_host(cards, limits):
    """Pre-run checks. Returns (problems, info)."""
    problems = []
    info = {"cards": []}
    if limits.get("require_cpu_performance", True):
        info["cpu_governors"] = govs = cpu_governors()
        if govs and govs != ["performance"]:
            problems.append(f"CPU governor is {govs}, expected performance")
    for card in cards:
        drm, name = card.get("drm"), f"card {card['index']} ({card.get('drm')})"
        if not drm:
            problems.append(f"{name}: no drm node configured")
            continue
        ci = {
            "index": card["index"],
            "drm": drm,
            "freq": read_freq(drm),
            "temps": read_temps(drm),
        }
        info["cards"].append(ci)
        if limits.get("require_idle", True):
            ci["busy_pids"] = card_busy_pids(drm)
            if ci["busy_pids"]:
                problems.append(f"{name} in use by pid(s) {ci['busy_pids']}")
        if limits.get("require_freq_lock", True):
            if not ci["freq"]:
                problems.append(f"{name}: no GPU frequency control found in sysfs")
            for d in ci["freq"]:
                if d["min"] is None or d["min"] != d["max"]:
                    problems.append(
                        f"{name}: GPU frequency not locked (min={d['min']} max={d['max']} at {d['dir']})"
                    )
        hot = {
            k: v
            for k, v in ci["temps"].items()
            if v > (limits.get("max_temp_c") or 1e9)
        }
        if hot:
            problems.append(f"{name}: temperature above {limits['max_temp_c']}C: {hot}")
    return problems, info


def run_env(args):
    cfg = machine_config(args.machines)
    cards = cfg["cards"]
    if args.action == "show":
        print(json.dumps(cfg, indent=2))
    elif args.action == "status":
        for c in cards:
            print(
                json.dumps(
                    {
                        "card": c["index"],
                        "freq": read_freq(c["drm"]),
                        "temps": read_temps(c["drm"]),
                    }
                )
            )
        print(json.dumps({"cpu_governors": cpu_governors()}))
    elif args.action == "lock":
        for c in cards:
            print(
                json.dumps(
                    {
                        "card": c["index"],
                        "locked": lock_gpu(c["drm"], args.ratio, args.mhz),
                    }
                )
            )
    elif args.action == "unlock":
        for c in cards:
            unlock_gpu(c["drm"])
    else:
        problems, info = check_host(cards, cfg.get("limits", {}))
        print(json.dumps(info, indent=2))
        for p in problems:
            print(f"ENV PROBLEM: {p}", file=sys.stderr)
        return 2 if problems else 0
    return 0


def add_env_parser(sub):
    p = sub.add_parser("env", help="machine layout, GPU frequency lock, pre-run checks")
    p.add_argument("action", choices=["show", "status", "lock", "unlock", "check"])
    p.add_argument(
        "--machines",
        default=None,
        help="default: the machines section of config/gating.yaml",
    )
    p.add_argument("--ratio", type=float, default=0.95, help="lock at ratio * RP0")
    p.add_argument("--mhz", type=int, default=None, help="lock at this frequency")
    p.set_defaults(func=lambda a: sys.exit(run_env(a)))


# ---------------------------------------------------------------------------
# Machine and card identity
# ---------------------------------------------------------------------------
# Reference records are matched per environment key (GPU model, kernel
# driver, user-mode driver, GPU firmware, frequency range) and per card.
# ``machine_id`` is the runner name (container hostnames change per job);
# ``card_id`` is the PCI address plus a serial number when sysfs has one.

FIRMWARE_FILES = (
    "device/tile0/gt0/uc/guc_version",
    "device/tile0/gt0/uc/huc_version",
    "device/gt/gt0/uc/guc_version",
    "device/vbios_version",
)
SERIAL_FILES = ("device/serial_number", "device/unique_id")


def machine_id():
    return (
        os.environ.get("MICROBENCH_MACHINE_ID")
        or os.environ.get("RUNNER_NAME")
        or socket.gethostname()
    )


def kmd_version():
    mods = [m for m in ("xe", "i915") if os.path.isdir(f"/sys/module/{m}")]
    parts = [platform.release()]
    for m in mods:
        parts.append(
            f"{m}:{_read(f'/sys/module/{m}/version') or _read(f'/sys/module/{m}/srcversion') or '-'}"
        )
    return " ".join(parts)


def _freq_text(doms):
    vals = {
        str(d["min"]) if d["min"] == d["max"] else f"{d['min']}-{d['max']}"
        for d in doms or []
        if d.get("min") and d.get("max")
    }
    return ",".join(sorted(vals)) or None


def card_identity(card):
    drm = card.get("drm")
    base = os.path.join(SYSFS, drm) if drm else None
    serial = firmware = None
    if base:
        serial = next(
            filter(None, (_read(os.path.join(base, f)) for f in SERIAL_FILES)), None
        )
        fw = [_read(os.path.join(base, f)) for f in FIRMWARE_FILES]
        firmware = ";".join(v.replace("\n", " ") for v in fw if v) or None
    bdf = card.get("bdf") or (
        os.path.basename(os.path.realpath(f"{base}/device")) if base else None
    )
    return {
        "card_id": "/".join(filter(None, (bdf, serial))) or f"index{card['index']}",
        "gpu_firmware": firmware,
        "freq_lock_mhz": _freq_text(read_freq(drm)) if drm else None,
    }


ENV_KEY_FIELDS = (
    "gpu_model",
    "kmd_version",
    "umd_version",
    "gpu_firmware",
    "freq_lock_mhz",
)


def env_key(fields):
    text = json.dumps([fields.get(k) for k in ENV_KEY_FIELDS])
    return hashlib.sha1(text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
# Sides are named wheels (venv pythons). Every (side, round) on a card is a
# fresh worker process pinned to the card (``ZE_AFFINITY_MASK``) and to the
# card's NUMA node / CPUs; all sides run the same benchmark code (this
# checkout), only the wheel differs.
#
# * A/B (``ab``/``aa``): rounds alternate ``A B | B A | A B ...``. Stage
#   ``screen`` (all cases), then ``confirm`` (screen-suspicious cases, B first,
#   moved to the next card, A and B on the same card).
# * single-sided (``single``): one side ``single``, N rounds (one process per
#   case each), case order shuffled between rounds.
# * multi-wheel (``multi``): any number of sides alternated per round
#   (``P P' N | N P' P``), used by the nightly attribution.
#
# Before the measurement: ``validate`` (``schedule --validate``). The scheduler records
# raw data, per-case logs (``logs/<op>/<pass>_<case_id>/``) and records
# (``records.jsonl``); verdicts are made elsewhere.

EXIT_OK, EXIT_INFRA, EXIT_VALIDATION = 0, 2, 3


class InfraError(Exception):
    pass


def lpt_shards(case_ids, costs, n):
    """Longest-processing-time-first packing of cases onto n cards."""
    shards, load = [[] for _ in range(n)], [0.0] * n
    for cid in sorted(case_ids, key=lambda c: -costs.get(c, DEFAULT_CASE_SECONDS)):
        i = load.index(min(load))
        shards[i].append(cid)
        load[i] += costs.get(cid, DEFAULT_CASE_SECONDS)
    return shards, load


def round_order(r, first="A", sides=("A", "B")):
    """Side order of round ``r``: ``first`` leads even rounds, odd rounds reverse."""
    order = list(sides)
    if first in order:
        i = order.index(first)
        order = order[i:] + order[:i]
    return order if r % 2 == 0 else order[::-1]


def torch_git_version(python):
    cmd = [python, "-c", "import torch; print(torch.version.git_version)"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise InfraError(f"{python}: cannot import torch: {out.stderr.strip()[-500:]}")
    return out.stdout.strip().splitlines()[-1]


def _selection_list(path, key):
    """Case ids from a JSON list or the ``key`` entry of selection.json."""
    data = load_json(path, []) if path else []
    return data.get(key, []) if isinstance(data, dict) else data


def runner_cards(args):
    """Cards of this runner, restricted to ``--cards``."""
    cards = machine_config(args.machines)["cards"]
    if args.cards:
        want = [int(c) for c in args.cards.split(",")]
        cards = [c for c in cards if c["index"] in want] or [{"index": i} for i in want]
    return cards or [{"index": 0}]


class Scheduler:
    """Runs workers over the cards. ``sides``: ordered {name: python}."""

    def __init__(self, args, sides, mode):
        self.args = args
        self.out = os.path.abspath(args.out_dir)
        self.raw = os.path.join(self.out, "raw")
        self.logs = os.path.join(self.out, "logs")
        os.makedirs(self.raw, exist_ok=True)
        self.machine = machine_config(args.machines)
        self.cards = runner_cards(args)
        self.python = dict(sides)
        self.mode = mode
        self.protocol = dict(self.machine.get("protocol", {}))
        self.protocol.update(load_json(args.protocol, {}))
        self.protocol_path = os.path.join(self.out, "protocol.json")
        with open(self.protocol_path, "w") as f:
            json.dump(self.protocol, f, indent=2)
        self.costs = load_history(args.history)["case_cost"]
        self.deadline = time.time() + args.budget_min * 60 if args.budget_min else None
        self.meta = {
            "mode": mode,
            "sides": list(self.python),
            "host": self.machine.get("host"),
            "machine_id": machine_id(),
            "kmd_version": kmd_version(),
            "protocol_version": PROTOCOL_VERSION,
            "cards": [
                dict(
                    {k: c.get(k) for k in ("index", "drm", "numa", "cpus")},
                    **card_identity(c),
                )
                for c in self.cards
            ],
            "python": self.python,
            "protocol": self.protocol,
            "stages": {},
            "infra_errors": [],
            "crashed_cases": [],
            "started": time.time(),
        }
        self.lock = threading.Lock()

    def _env(self, card):
        e = dict(os.environ)
        e["PYTHONPATH"] = str(PACKAGE_DIR.parent) + os.pathsep + e.get("PYTHONPATH", "")
        if self.args.device == "xpu":
            e["ZE_AFFINITY_MASK"] = str(card["index"])
            e["ZE_ENABLE_TRACING_LAYER"] = "1"
        e.setdefault("OMP_NUM_THREADS", "4")
        return e

    def _spawn(self, side, card, argv, prefix=(), **kw):
        """``python -m microbench <argv>`` with ``side``'s python on ``card``."""
        cmd = [*prefix, self.python[side], "-m", "microbench", *argv]
        cmd += ["--device", self.args.device, "--config", self.protocol_path]
        return subprocess.run(
            cmd, env=self._env(card), cwd=str(PACKAGE_DIR.parent), **kw
        )

    def run_worker(self, side, card, stage, rnd, case_ids):
        """Run one worker; restart after a crash, skipping the crashing case."""
        tag = f"{stage}_c{card['index']}_r{rnd}_{side}"
        out = os.path.join(self.raw, f"{tag}.jsonl")
        mine = dict(side=side, round=rnd, card=str(card["index"]), stage=stage)
        remaining, crashes = list(case_ids), 0
        while remaining:
            case_file = os.path.join(self.raw, f"{tag}.cases.{crashes}.json")
            with open(case_file, "w") as f:
                json.dump(remaining, f)
            argv = ["worker", "--cases", case_file, "--out", out, "--side", side]
            argv += ["--round", str(rnd), "--card", mine["card"], "--stage", stage]
            argv += ["--log-dir", self.logs]
            argv += ["--quick"] if self.args.quick else []
            argv += ["--keep-traces"] if self.args.keep_traces else []
            prefix = bind_prefix(card, self.args.binding)
            with open(os.path.join(self.raw, f"{tag}.log"), "a") as log:
                try:
                    timeout = 300 + self.args.case_timeout_s * len(remaining)
                    proc = self._spawn(
                        side,
                        card,
                        argv,
                        prefix,
                        stdout=log,
                        stderr=log,
                        timeout=timeout,
                    )
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            recs = [
                r
                for r in read_jsonl(out)
                if r.get("round") == rnd and r.get("side") == side
            ]
            done = {r["case_id"] for r in recs if r.get("type") == "result"}
            started = [r["case_id"] for r in recs if r.get("type") == "start"]
            remaining = [c for c in remaining if c not in done]
            if rc == 0 and not remaining:
                return
            crashed = started[-1] if started and started[-1] in remaining else None
            if crashed is None or crashes >= self.args.max_restarts:
                with self.lock:
                    self.meta["infra_errors"].append(
                        f"worker {tag} exited with {rc}; {len(remaining)} case(s) not measured"
                    )
                return
            crashes += 1
            with self.lock:
                self.meta["crashed_cases"].append(
                    {"case_id": crashed, "worker": tag, "rc": rc}
                )
            with open(out, "a") as f:
                rec = dict(
                    mine,
                    type="result",
                    case_id=crashed,
                    status="crashed",
                    rc=str(rc),
                    log=f"raw/{tag}.log",
                )
                f.write(json.dumps(rec) + "\n")
            remaining.remove(crashed)

    def run_stage(
        self,
        stage,
        shards,
        rounds,
        first=None,
        shuffle=False,
        start_round=0,
    ):
        sides = list(self.python)
        first = first or sides[0]
        t0 = time.time()
        info = self.meta["stages"].setdefault(stage, {})

        def card_loop(card, ids):
            for r in range(start_round, start_round + rounds):
                if self.deadline and time.time() > self.deadline:
                    with self.lock:
                        info["truncated_at_round"] = r
                    return
                order = list(ids)
                if shuffle:
                    random.Random(f"{stage}/{r}/{card['index']}").shuffle(order)
                for side in round_order(r, first, sides):
                    self.run_worker(side, card, stage, r, order)

        threads = [
            threading.Thread(target=card_loop, args=(card, ids))
            for card, ids in zip(self.cards, shards)
            if ids
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        info.update(
            seconds=round(info.get("seconds", 0) + time.time() - t0, 1),
            rounds=start_round + rounds,
            first=first,
            sides=sides,
        )
        shard_map = info.setdefault("shards", {})
        for c, ids in zip(self.cards, shards):
            shard_map.setdefault(str(c["index"]), [])
            shard_map[str(c["index"])] += [
                i for i in ids if i not in shard_map[str(c["index"])]
            ]

    def check_versions(self, same=None):
        """PyTorch SHA of every side; ``same`` lists sides that must agree."""
        shas = {side: torch_git_version(py) for side, py in self.python.items()}
        self.meta["torch_git_version"] = shas
        same = (
            same
            if same is not None
            else (list(shas) if not self.args.allow_sha_mismatch else [])
        )
        if len({shas[s] for s in same}) > 1:
            raise InfraError(
                "PyTorch SHA differs between wheels: "
                + ", ".join(f"{s}={shas[s]}" for s in same)
            )
        want = self.args.expect_torch_sha
        last = list(shas)[-1]
        if want and not (
            shas[last].startswith(want[:12]) or want.startswith(shas[last][:12])
        ):
            raise InfraError(f"PyTorch SHA {shas[last]} != resolved SHA {want}")

    def snapshot_env(self, key):
        self.meta[key] = {
            str(c["index"]): {
                "freq": read_freq(c["drm"]),
                "temps": read_temps(c["drm"]),
            }
            for c in self.cards
            if c.get("drm")
        }

    def validate(self, ids):
        """Validate ``ids`` with one validator per card, card ``k`` taking every
        n-th case (``ids`` is in priority order). A validator that crashes
        (e.g. the device is lost) fails the case it was on and restarts on the
        cases after it."""
        side = list(self.python)[-1]
        budget = self.args.validate_budget_min
        deadline = time.time() + budget * 60 if budget else None
        n = min(len(self.cards), len(ids))
        cases = load_cases()
        results, errors = [[] for _ in range(n)], []

        def one(k):
            card = self.cards[k]
            tag = f"validate_c{card['index']}"
            log = os.path.join(self.raw, f"{tag}.log")
            remaining, crashes = ids[k::n], 0
            while remaining:
                case_file = os.path.join(self.raw, f"{tag}.cases.{crashes}.json")
                with open(case_file, "w") as f:
                    json.dump(remaining, f)
                out = os.path.join(self.out, f"validation.{tag}.jsonl")
                left = max(0.01, (deadline - time.time()) / 60) if deadline else 0
                cmd = ["validate", "--cases", case_file, "--out", out]
                cmd += ["--budget-min", str(left)]
                start = os.path.getsize(log) if os.path.exists(log) else 0
                try:
                    with open(log, "a") as fh:
                        rc = self._spawn(
                            side,
                            card,
                            cmd,
                            stdout=fh,
                            stderr=fh,
                            # the budget is checked between cases; allow one
                            # slow case on top
                            timeout=left * 60 + 600 if deadline else None,
                        ).returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
                got = read_jsonl(out)
                if os.path.exists(out):
                    os.remove(out)
                results[k] += got
                done = {r["case_id"] for r in got}
                remaining = [c for c in remaining if c not in done]
                if not remaining:
                    return
                if rc == 0 or crashes >= self.args.max_restarts:
                    with self.lock:
                        errors.append(
                            f"validator on card {card['index']} exited with {rc}; "
                            f"{len(remaining)} case(s) not validated"
                        )
                    return
                crashes += 1
                # results are written in order: the first missing case crashed
                cid = remaining.pop(0)
                with open(log, errors="replace") as fh:
                    fh.seek(start)
                    lines = [ln.strip() for ln in fh if "Error" in ln]
                why = lines[-1] if lines else f"exit {rc}"
                case = cases.get(cid)
                results[k].append(
                    {
                        "status": "fail",
                        "failures": [
                            f"validator crashed on this case: {why} (raw/{tag}.log)"
                        ],
                        "case_id": cid,
                        "op": case.op if case else None,
                        "pass_": case.pass_ if case else None,
                        "label": case.label if case else None,
                    }
                )
                with self.lock:
                    self.meta["crashed_cases"].append(
                        {"case_id": cid, "worker": tag, "rc": rc}
                    )

        threads = [threading.Thread(target=one, args=(k,)) for k in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = [r for part in results for r in part]
        with open(os.path.join(self.out, "validation.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        failed = [r for r in results if r.get("status") == "fail"]
        keys = ("case_id", "op", "pass_", "label", "failures")
        self.meta["validation"] = {
            "cases": len(ids),
            "cards": n,
            "failed": [{k: r.get(k) for k in keys} for r in failed],
            "not_validated": [
                r["case_id"] for r in results if r.get("status") == "not_validated"
            ],
        }
        if errors:
            raise InfraError("; ".join(errors))
        return failed

    def suspicious(self, ids, stage="screen"):
        recs = []
        for name in os.listdir(self.raw):
            if name.startswith(f"{stage}_") and name.endswith(".jsonl"):
                recs += read_jsonl(os.path.join(self.raw, name))
        groups = judge.group_samples(
            recs, "e2e_us" if self.args.device == "cpu" else "device_us"
        )
        sus = []
        for cid in ids:
            g = groups.get((cid, stage))
            if g and g.get("A") and g.get("B"):
                res = judge.compare(g["A"], g["B"], n_boot=500)
                if judge.suspicious(res) or judge.classify(res)[0] in (
                    "regression",
                    "improvement",
                ):
                    sus.append(cid)
        return sus

    def prepare(self, validate_ids=(), same_sha=None):
        """Versions, environment check, validation. Raises InfraError; returns
        EXIT_VALIDATION when a changed case fails validation."""
        self.check_versions(same_sha)
        if self.args.env_check:
            problems, self.meta["env_check"] = check_host(
                [c for c in self.cards if c.get("drm")],
                self.machine.get("limits", {}),
            )
            if problems:
                raise InfraError("environment check failed: " + "; ".join(problems))
        self.snapshot_env("env_start")
        if validate_ids and self.validate(list(validate_ids)):
            return EXIT_VALIDATION
        return EXIT_OK

    def measure_single(self, ids, rounds, stage="single", start_round=0):
        shards, load = lpt_shards(ids, self.costs, len(self.cards))
        self.meta.setdefault("estimated_load_s", [round(x, 1) for x in load])
        self.run_stage(stage, shards, rounds, shuffle=True, start_round=start_round)

    def measure_ab(self, ids, screen_rounds, confirm_rounds):
        n = len(self.cards)
        shards, load = lpt_shards(ids, self.costs, n)
        self.meta["estimated_load_s"] = [round(x, 1) for x in load]
        self.run_stage("screen", shards, screen_rounds)
        if confirm_rounds <= 0:
            return
        if self.deadline and time.time() > self.deadline:
            self.meta["confirm_skipped"] = "time budget exhausted"
            return
        sus = self.meta["suspicious"] = self.suspicious(ids)
        # each suspicious case moves to the next card (A and B together)
        cshards = [[] for _ in range(n)]
        for k, shard in enumerate(shards):
            cshards[(k + 1) % n] += [c for c in shard if c in sus]
        if any(cshards):
            self.run_stage("confirm", cshards, confirm_rounds, first="B")

    def measure_multi(self, ids, rounds, stage="attribution"):
        shards, load = lpt_shards(ids, self.costs, len(self.cards))
        self.meta["estimated_load_s"] = [round(x, 1) for x in load]
        self.run_stage(stage, shards, rounds)

    def finish(self, status, side_info=None):
        """Write run_meta.json, records.jsonl and the per-case logs."""
        try:
            self.snapshot_env("env_end")
        except OSError:
            pass
        if status == EXIT_OK and self.meta["infra_errors"]:
            status = EXIT_INFRA
        self.meta.update(finished=time.time(), exit_status=status)
        recs = judge.load_records(self.raw)
        records = case_records(recs, self.meta, side_info or {}, self.args.run_url)
        with open(os.path.join(self.out, "records.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        write_case_logs(self.out, recs, records)
        with open(os.path.join(self.out, "run_meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)
        return status

    def run(self):
        """CLI flow of ``python -m microbench schedule``."""
        a = self.args
        ids = _selection_list(a.cases, "cases")
        self.meta["selected"] = len(ids)
        status = EXIT_OK
        try:
            same = [] if self.mode == "multi" else None
            status = self.prepare(_selection_list(a.validate, "validate"), same)
            if status == EXIT_OK and ids:
                if self.mode == "single":
                    self.measure_single(ids, a.screen_rounds)
                elif self.mode == "multi":
                    self.measure_multi(ids, a.confirm_rounds)
                else:
                    self.measure_ab(ids, a.screen_rounds, a.confirm_rounds)
        except InfraError as e:
            self.meta["infra_errors"].append(str(e))
            status = EXIT_INFRA
        return self.finish(status, load_json(a.side_info, {}))


# ---------------------------------------------------------------------------
# Records and per-case logs
# ---------------------------------------------------------------------------


def _stats(vals):
    vals = [v for v in vals or [] if v is not None]
    if not vals:
        return None, None, None, 0
    return (
        round(statistics.median(vals), 3),
        round(min(vals), 3),
        round(max(vals), 3),
        len(vals),
    )


def card_env(meta, recs):
    """{card: env fields + env_key}: card identity plus the worker's driver info."""
    workers = {}
    for r in recs:
        if r.get("type") == "env":
            workers.setdefault(str(r.get("card")), r)
    out = {}
    for c in meta.get("cards", []):
        w = workers.get(str(c["index"]), {})
        fields = {
            "gpu_model": w.get("device_name"),
            "kmd_version": meta.get("kmd_version"),
            "umd_version": w.get("driver_version"),
            "gpu_firmware": c.get("gpu_firmware"),
            "freq_lock_mhz": c.get("freq_lock_mhz"),
        }
        out[str(c["index"])] = dict(
            fields, card_id=c.get("card_id"), env_key=env_key(fields)
        )
    return out


def case_records(recs, meta, side_info, run_url=None):
    """One record per (case, side, stage): case, commits, card, environment
    and per-process statistics."""
    envs = card_env(meta, recs)
    groups = {}
    for r in recs:
        if r.get("type") == "result":
            groups.setdefault((r["case_id"], r.get("side"), r.get("stage")), []).append(
                r
            )
    try:
        manifest = load_cases()
    except Exception:  # noqa: BLE001 - records without case metadata
        manifest = {}
    shas = meta.get("torch_git_version") or {}
    out = []
    for (cid, side, stage), rs in sorted(
        groups.items(), key=lambda kv: tuple(map(str, kv[0]))
    ):
        card = statistics.mode(str(r.get("card")) for r in rs)
        info = side_info.get(side) or {}
        case = manifest.get(cid)
        procs = []
        for r in sorted(rs, key=lambda r: r.get("round", 0)):
            dev = (
                _stats(r.get("device_us"))
                if r.get("device_status") in ("ok", None)
                else (None,) * 3 + (0,)
            )
            e2e = _stats(r.get("e2e_us"))
            procs.append(
                {
                    "round": r.get("round"),
                    "card": str(r.get("card")),
                    "status": r.get("status"),
                    "device_status": r.get("device_status"),
                    "device_us": dev[0],
                    "device_min": dev[1],
                    "device_max": dev[2],
                    "n_device": dev[3],
                    "e2e_us": e2e[0],
                    "e2e_min": e2e[1],
                    "e2e_max": e2e[2],
                    "n_e2e": e2e[3],
                    "error": r.get("error"),
                }
            )
        env = envs.get(card, {})
        out.append(
            {
                "case_id": cid,
                "op": case.op if case else None,
                "pass": case.pass_ if case else None,
                "case_code_hash": case.case_code_hash if case else None,
                "protocol_version": PROTOCOL_VERSION,
                "side": side,
                "stage": stage,
                "role": info.get("role"),
                "xpu_ops_sha": info.get("xpu_ops_sha"),
                "pytorch_sha": shas.get(side),
                "machine_id": meta.get("machine_id"),
                "card": card,
                **env,
                "per_process": procs,
                "timestamp": time.time(),
                "run_url": run_url,
            }
        )
    return out


def process_medians(record, metric="device_us"):
    return [
        p[metric] for p in record.get("per_process") or [] if p.get(metric) is not None
    ]


def _fmt(x):
    return "-" if x is None else f"{x:.3f}"


def find_case_dir(logs_dir, case_id):
    """``logs/<op>/<pass>_<case_id>`` of a case, if it has logs."""
    found = glob.glob(os.path.join(logs_dir, "*", f"*_{case_id}"))
    return found[0] if found else None


def write_case_logs(out_dir, recs, records):
    """``<op>_<pass>_samples.jsonl`` and ``<op>_<pass>_summary.log`` of every
    case, and ``logs/index.csv`` (case_id -> op, pass, params, directory)."""
    try:
        cases = load_cases()
    except Exception:  # noqa: BLE001 - logs without case metadata
        cases = {}
    index = []
    by_case = {}
    for r in recs:
        if r.get("type") == "result":
            by_case.setdefault(r["case_id"], []).append(r)
    rec_by_case = {}
    for rec in records:
        rec_by_case.setdefault(rec["case_id"], []).append(rec)
    for cid, rs in by_case.items():
        case = cases.get(cid)
        op, pass_ = (case.op, case.pass_) if case else ("unknown", "unknown")
        rel = case_log_dir(op, pass_, cid)
        index.append([cid, op, pass_, case.label if case else "", rel])
        d = os.path.join(out_dir, "logs", rel)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, case_log_name(op, pass_, "samples.jsonl")), "w") as f:
            # device time: one sample per iteration; E2E: one per pass
            for r in rs:
                where = {
                    "side": r.get("side"),
                    "stage": r.get("stage"),
                    "process": r.get("round"),
                    "card": r.get("card"),
                }
                ok = r.get("device_status") in ("ok", None)
                for i, dv in enumerate(r.get("device_us") or []):
                    line = dict(
                        where, iter=i, device_us=dv, valid=dv is not None and ok
                    )
                    f.write(json.dumps(line) + "\n")
                for k, e2e in enumerate(r.get("e2e_us") or []):
                    f.write(
                        json.dumps(dict(where, **{"pass": k, "e2e_us": e2e})) + "\n"
                    )
        case = cases.get(cid)
        lines = [f"case {cid}"]
        if case:
            p = case.params
            lines += [
                f"op {case.op}  pass {case.pass_}  tier {case.tier}",
                f"shape {p.get('shape', p.get('shapes'))}  dtype {p.get('dtype')}  "
                f"layout {'channels_last' if p.get('channels_last') else 'strided'}",
                f"params {case.label}",
                f"repro: {case.repro()}",
            ]
        for rec in sorted(
            rec_by_case.get(cid, []), key=lambda x: (str(x["stage"]), str(x["side"]))
        ):
            lines += [
                "",
                f"[{rec['stage']}] side {rec['side']} role {rec.get('role')} xpu-ops {rec.get('xpu_ops_sha')} "
                f"PyTorch {rec.get('pytorch_sha')} card {rec['card']} ({rec.get('card_id')})",
                "  proc status   device med/min/max (n)            e2e med/min/max (n)",
            ]
            for pp in rec["per_process"]:
                lines.append(
                    f"  {pp['round']:>4} {str(pp['status']):8s} "
                    f"{_fmt(pp['device_us'])}/{_fmt(pp['device_min'])}/{_fmt(pp['device_max'])} ({pp['n_device']})   "
                    f"{_fmt(pp['e2e_us'])}/{_fmt(pp['e2e_min'])}/{_fmt(pp['e2e_max'])} ({pp['n_e2e']})"
                )
                if pp.get("error"):
                    lines.append(f"       error: {pp['error']}")
            for metric in ("device_us", "e2e_us"):
                meds = process_medians(rec, metric)
                if meds:
                    lines.append(
                        f"  median of process medians ({metric}): {statistics.median(meds):.3f}"
                    )
        for r in rs:
            if r.get("status") not in ("ok", None):
                lines += [
                    "",
                    f"[{r.get('stage')}] side {r.get('side')} process {r.get('round')}: {r.get('status')}",
                ]
                if r.get("error"):
                    lines.append(f"  error: {r['error']}")
                if r.get("traceback"):
                    lines += ["  " + t for t in r["traceback"].splitlines()]
                log = r.get("log")
                if log and os.path.exists(os.path.join(out_dir, log)):
                    with open(os.path.join(out_dir, log), errors="replace") as f:
                        tail = f.read().splitlines()[-40:]
                    lines += [f"  worker stderr ({log}, last {len(tail)} lines):"] + [
                        "    " + t for t in tail
                    ]
        # A/B comparisons per stage, when both sides exist
        stages = {}
        for rec in rec_by_case.get(cid, []):
            stages.setdefault(rec["stage"], {})[rec["side"]] = rec
        groups = {m: judge.group_samples(rs, m) for m in ("device_us", "e2e_us")}
        for stage, sides in sorted(stages.items()):
            if "A" in sides and "B" in sides:
                for m, g in groups.items():
                    gg = g.get((cid, stage))
                    res = (
                        judge.compare(gg["A"], gg["B"], n_boot=500)
                        if gg and gg["A"] and gg["B"]
                        else None
                    )
                    if res:
                        lines.append(
                            f"[{stage}] {m}: A {res['median_a']:.3f}  B {res['median_b']:.3f}  "
                            f"ratio {res['ratio']:.4f}  95% CI [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}]"
                        )
        with open(os.path.join(d, case_log_name(op, pass_, "summary.log")), "w") as f:
            f.write("\n".join(lines) + "\n")
    if index:
        with open(os.path.join(out_dir, "logs", "index.csv"), "w", newline="") as f:
            csv.writer(f).writerows(
                [["case_id", "op", "pass", "params", "dir"]]
                + sorted(index, key=lambda r: r[4])
            )


def append_case_summary(out_dir, case_id, lines):
    """Add conclusion lines (detection / verdict) to the case's summary log."""
    d = find_case_dir(os.path.join(out_dir, "logs"), case_id)
    summary = glob.glob(os.path.join(d, "*_summary.log")) if d else []
    if summary:
        with open(summary[0], "a") as f:
            f.write("\n".join(["", *lines]) + "\n")


def summarize_exit(meta, status):
    if status == EXIT_INFRA:
        lines = ["### Benchmark did not complete (infra error)", ""]
        lines += [f"- {e}" for e in meta.get("infra_errors", [])]
        lines += ["", "Please rerun the failed job."]
    elif status == EXIT_VALIDATION:
        lines = [
            "### Microbench case validation failed",
            "",
            "The changed cases below do not measure the op they declare. Fix the case "
            "definitions; the data of a failing case cannot be used.",
            "",
        ]
        for r in meta.get("validation", {}).get("failed", []):
            lines.append(
                f"- `{r.get('op')}/{r.get('pass_')}` `{r.get('label')}` (`{r.get('case_id')}`)"
            )
            lines += [f"  - {f}" for f in r.get("failures") or []]
    else:
        stages = ", ".join(
            f"{k} {v.get('seconds')}s" for k, v in meta.get("stages", {}).items()
        )
        lines = [
            f"Microbench run finished: mode `{meta.get('mode')}`, {meta.get('selected', 0)} case(s), stages: {stages}"
        ]
        skipped = meta.get("validation", {}).get("not_validated") or []
        if skipped:
            lines.append(
                f"Validation time budget exhausted: {len(skipped)} changed case(s) were not validated (report only)."
            )
    return "\n".join(lines) + "\n"


def scheduler_mode(args):
    """(sides, mode) from the schedule CLI arguments."""
    if args.wheel:
        sides = dict(w.split("=", 1) for w in args.wheel)
        return sides, "multi"
    if args.single:
        return {"single": args.python_b}, "single"
    a = args.python_a or args.python_b
    same = os.path.realpath(a) == os.path.realpath(args.python_b)
    return {"A": a, "B": args.python_b}, "aa" if same else "ab"


def run_schedule(args):
    sched = Scheduler(args, *scheduler_mode(args))
    status = sched.run()
    text = summarize_exit(sched.meta, status)
    print(text)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    return status


def add_scheduler_args(p):
    """Arguments shared by ``schedule`` and ``ci-run``."""
    p.add_argument(
        "--validate-budget-min",
        type=float,
        default=20.0,
        help="validation time limit; later cases are reported as not validated",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="xpu")
    p.add_argument("--cards", default=None, help="comma separated card indices")
    p.add_argument("--machines", default=None)
    p.add_argument("--protocol", default=None, help="protocol overrides JSON")
    p.add_argument("--history", default=None, help="default: config/history.json")
    p.add_argument(
        "--binding", default="numactl", choices=["numactl", "taskset", "none"]
    )
    p.add_argument("--budget-min", type=float, default=30.0)
    p.add_argument(
        "--case-timeout-s", type=float, default=180.0, help="worker timeout per case"
    )
    p.add_argument(
        "--max-restarts",
        type=int,
        default=2,
        help="crashing cases tolerated per worker",
    )
    p.add_argument("--expect-torch-sha", default=None)
    p.add_argument(
        "--allow-sha-mismatch", action="store_true", help="sides may differ in PyTorch"
    )
    p.add_argument(
        "--env-check",
        action="store_true",
        help="fail on busy card / unlocked freq / hot GPU",
    )
    p.add_argument(
        "--quick", action="store_true", help="smoke mode: 3 iterations, no warmup"
    )
    p.add_argument(
        "--keep-traces",
        action="store_true",
        help="keep the device-pass chrome trace of every case (debug, large)",
    )
    p.add_argument("--run-url", default=None, help="recorded in records.jsonl")
    p.add_argument(
        "--summary",
        default=None,
        help="append a markdown summary (GITHUB_STEP_SUMMARY)",
    )


def add_schedule_parser(sub):
    p = sub.add_parser(
        "schedule", help="run A/B, A/A, single-sided or multi-wheel measurement"
    )
    p.add_argument(
        "--python-a", default=None, help="side A (base wheel); default: B (A/A)"
    )
    p.add_argument("--python-b", default=sys.executable, help="side B (PR wheel)")
    p.add_argument(
        "--single", action="store_true", help="single-sided: only the B wheel"
    )
    p.add_argument(
        "--wheel",
        action="append",
        default=[],
        help="NAME=PYTHON, repeatable: multi-wheel alternation in this order",
    )
    p.add_argument(
        "--side-info",
        default=None,
        help='JSON {side: {"xpu_ops_sha": ..., "role": ...}} for records.jsonl',
    )
    p.add_argument(
        "--cases", required=True, help="JSON list of case_ids or selection.json"
    )
    p.add_argument(
        "--validate", default=None, help="cases to validate first (same format)"
    )
    p.add_argument("--screen-rounds", type=int, default=3)
    p.add_argument("--confirm-rounds", type=int, default=6)
    add_scheduler_args(p)
    p.set_defaults(func=lambda a: sys.exit(run_schedule(a)))


# ---------------------------------------------------------------------------
# CI pipeline: python -m microbench ci-run --scope <scope>
# ---------------------------------------------------------------------------
# ``python -m microbench ci-run --scope <scope>``: the microbench-run job.
#
# | scope                        | wheels           | what                                              |
# |------------------------------|------------------|---------------------------------------------------|
# | pr-vs-base                   | base, B          | on-demand A/B of the selected cases               |
# | nightly-vs-reference         | N (+ P)          | one shard: measurement, A/A rotation, detection   |
# |                              |                  | recheck vs P, candidate commits                   |
# | nightly-xpu-ops-vs-pytorch   | P, P', N         | three-wheel alternation for attribution           |
# | ondemand                     | B                | two-stage A/A of one wheel (report only)          |
#
# Every scope writes ``--out-dir``: ``run_meta.json``, ``raw/``, ``logs/``,
# ``records.jsonl`` and its result file (``ab_cases.json``, ``nightly.json``);
# sub-runs (A/A, recheck) live in sub-directories. The verdicts that need
# several runs are made by ``python -m microbench check``.
# Exit codes: 0 ok, 2 infrastructure error.

SCOPES = (
    "pr-vs-base",
    "nightly-vs-reference",
    "nightly-xpu-ops-vs-pytorch",
    "ondemand",
)


def _sub_args(args, out_dir):
    a = copy.copy(args)
    a.out_dir = out_dir
    return a


def _metric(args):
    return "e2e_us" if args.device == "cpu" else "device_us"


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


def _measure(sched, ids, same_sha, body):
    """prepare, then ``body(sched)``; returns the exit status."""
    status = EXIT_OK
    try:
        status = sched.prepare(same_sha=same_sha)
        if status == EXIT_OK and ids:
            body(sched)
    except InfraError as e:
        sched.meta["infra_errors"].append(str(e))
        status = EXIT_INFRA
    return status


def _detect_lines(res):
    line = f"detection: {res['status']}"
    if res.get("ratio") is not None:
        line += f", ratio {res['ratio']:.4f} vs reference {res['reference_us']:.3f} us (threshold {res['threshold']:.3f})"
    if res.get("reason"):
        line += f" ({res['reason']})"
    return [line]


# ---------------------------------------------------------------------------
# On-demand scopes
# ---------------------------------------------------------------------------


def run_pr_vs_base(args, sel, cfg):
    ids = sel.get("cases") or []
    side_info = {
        "A": {"xpu_ops_sha": args.base_sha, "role": "pr_vs_base_a"},
        "B": {"xpu_ops_sha": args.head_sha, "role": "pr_vs_base_b"},
    }
    sched = Scheduler(args, {"A": args.python_base, "B": args.python}, "ab")
    sched.meta["selected"] = len(ids)
    status = _measure(sched, ids, None, lambda s: s.measure_ab(ids, 3, 6))
    status = sched.finish(status, side_info)
    _write_json(
        os.path.join(args.out_dir, "ab_cases.json"),
        {"scope": "pr-vs-base", "path": sel.get("path"), "cases": ids},
    )
    return status, summarize_exit(sched.meta, status), {}


def run_ondemand(args, sel, cfg):
    ids = sel.get("cases") or []
    side_info = {
        s: {"xpu_ops_sha": args.head_sha, "role": "ondemand_aa"} for s in ("A", "B")
    }
    sched = Scheduler(args, {"A": args.python, "B": args.python}, "aa")
    sched.meta["selected"] = len(ids)
    status = _measure(sched, ids, None, lambda s: s.measure_ab(ids, 3, 6))
    status = sched.finish(status, side_info)
    _write_json(
        os.path.join(args.out_dir, "ab_cases.json"),
        {"scope": "ondemand", "path": "aa", "cases": ids},
    )
    return status, summarize_exit(sched.meta, status), {}


# ---------------------------------------------------------------------------
# Nightly scopes
# ---------------------------------------------------------------------------


def aa_rotation(ids, man, fraction, last_aa=None):
    """PR-tier cases measured as A/A tonight: the slice after the one of the
    previous nightly A/A (``last_aa``, one of its case_ids), so 1/fraction
    nightlies cover every PR-tier case whatever days the nightlies run."""
    period = max(1, round(1 / fraction))
    k = 0 if last_aa is None else (int(last_aa, 16) + 1) % period
    return [
        c
        for c in ids
        if man.get(c, {}).get("tier") == "pr" and int(c, 16) % period == k
    ]


def kernel_names(log_dir, case_id, limit=12):
    """Device kernels counted for a case (from its device logs), most frequent first."""
    d = find_case_dir(log_dir, case_id)
    counts = {}
    if d:
        for name in sorted(os.listdir(d)):
            if name.endswith("_device.log"):
                with open(os.path.join(d, name), errors="replace") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 4 and parts[0] == "counted":
                            k = " ".join(parts[2:-1])
                            counts[k] = counts.get(k, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:limit]]


def run_nightly_vs_reference(args, sel, cfg):
    screening, metric = cfg.screening, _metric(args)
    det = screening["nightly_detect"]
    gating, hist = cfg.gating, load_history(args.history)
    man = {m["case_id"]: m for m in build_manifest()}
    shards, _ = lpt_shards(sel.get("cases") or [], cfg.costs, max(1, args.shards))
    ids = shards[args.shard] if args.shard < len(shards) else []
    state = reference.load_state(args.state)
    jumps = reference.last_jumps(state)
    entries = reference.by_case(reference.nightly_entries(state))
    noise = reference.noise_stats(state, screening["default_noise"], jumps)

    side_info = {
        "single": {"xpu_ops_sha": args.head_sha, "role": "nightly_vs_reference"}
    }
    sched = Scheduler(args, {"single": args.python}, "single")
    sched.meta.update(selected=len(ids), shard=args.shard, shards=args.shards)
    status = _measure(sched, ids, [], lambda s: s.measure_single(ids, det["processes"]))
    status = sched.finish(status, side_info)
    doc = {
        "scope": "nightly-vs-reference",
        "shard": args.shard,
        "status": status,
        "metric": metric,
        "cur": {"run_id": args.run_id, "xpu_ops_sha": args.head_sha},
        "prev": {"run_id": args.prev_run_id, "xpu_ops_sha": args.prev_sha},
        "state_run": args.state_run,
        "selected_ids": ids,
    }
    if status != EXIT_OK:
        _write_json(os.path.join(args.out_dir, "nightly.json"), doc)
        return status, summarize_exit(sched.meta, status), {}
    doc["cur"]["pytorch_sha"] = (sched.meta.get("torch_git_version") or {}).get(
        "single"
    )

    # A/A rotation (sigma_ab, delta), reference data only
    aa_ids = aa_rotation(ids, man, det["aa_fraction"], reference.last_aa_case(state))
    if aa_ids and not (sched.deadline and time.time() > sched.deadline):
        aa = Scheduler(
            _sub_args(args, os.path.join(args.out_dir, "aa")),
            {"A": args.python, "B": args.python},
            "aa",
        )
        aa.deadline = sched.deadline
        st = _measure(aa, aa_ids, None, lambda s: s.measure_ab(aa_ids, 3, 0))
        aa.finish(
            st,
            {
                s: {"xpu_ops_sha": args.head_sha, "role": "nightly_aa"}
                for s in ("A", "B")
            },
        )
        doc["aa_cases"] = aa_ids

    # detection
    detect = {}
    for rec in case_records(judge.load_records(sched.raw), sched.meta, side_info):
        cid = rec["case_id"]
        if all(p["status"] != "ok" for p in rec["per_process"]):
            detect[cid] = {"case_id": cid, "status": "error", "reason": "no result"}
            continue
        cands = reference.eligible_entries(
            entries.get(cid, []), cid, rec.get("case_code_hash"), jumps
        )
        detect[cid] = reference.classify(
            rec, cands, noise.get(cid), screening, metric=metric
        )
        append_case_summary(args.out_dir, cid, _detect_lines(detect[cid]))
    doc["detect"] = detect

    # recheck: previous nightly wheel (A) vs this one (B)
    recheck = [
        c
        for c, r in detect.items()
        if r["status"] in ("suspect_regression", "suspect_improvement")
    ]
    missing = [
        c
        for c, r in detect.items()
        if r["status"] in ("no_reference", "screen_invalid")
    ]
    if len(missing) > det["max_no_reference_recheck"]:
        core = set(cfg.core_cases())
        doc["no_reference_not_rechecked"] = [
            c for c in missing if man.get(c, {}).get("tier") != "pr" and c not in core
        ]
        missing = [
            c for c in missing if c not in set(doc["no_reference_not_rechecked"])
        ]
    recheck += missing
    doc["recheck_cases"] = recheck
    verdicts = {}
    if recheck and not args.python_prev:
        doc["recheck_skipped"] = "no previous nightly wheel"
    elif recheck:
        rdir = os.path.join(args.out_dir, "recheck")
        rargs = _sub_args(args, rdir)
        rargs.budget_min = args.recheck_budget_min
        rc = Scheduler(rargs, {"A": args.python_prev, "B": args.python}, "ab")
        st = _measure(rc, recheck, [], lambda s: s.measure_ab(recheck, 3, 6))
        st = rc.finish(
            st,
            {
                "A": {"xpu_ops_sha": args.prev_sha, "role": "recheck"},
                "B": {"xpu_ops_sha": args.head_sha, "role": "recheck"},
            },
        )
        doc["prev"]["pytorch_sha"] = (rc.meta.get("torch_git_version") or {}).get("A")
        if st != EXIT_OK:
            doc["recheck_skipped"] = "; ".join(rc.meta["infra_errors"]) or f"exit {st}"
        else:
            res = judge.judge(
                rc.raw,
                man,
                rc.meta,
                selection={"cases": recheck},
                gating=gating,
                noise=judge.merge_noise(hist["noise_stats"], noise),
            )
            for row in res["rows"]:
                m = row["metrics"].get("device") or row["metrics"].get("e2e") or {}
                used = m.get("used") or {}
                verdicts[row["case_id"]] = {
                    "verdict": row.get("verdict") or row["status"],
                    "reason": m.get("reason") or row.get("detail", ""),
                    "before_us": used.get("median_a"),
                    "after_us": used.get("median_b"),
                    "ratio": used.get("ratio"),
                    "ci": [used.get("ci_lo"), used.get("ci_hi")],
                }
                append_case_summary(
                    rdir,
                    row["case_id"],
                    [f"recheck verdict: {verdicts[row['case_id']]}"],
                )
    doc["recheck"] = verdicts

    # candidate commits
    candidates, kernels = {}, {}
    for cid, v in verdicts.items():
        if v["verdict"] == "regression":
            candidates[cid] = (
                candidate_commits(
                    args.repo, args.prev_sha, args.head_sha, man[cid]["op"], cfg.srcmap
                )
                if args.prev_sha and args.head_sha
                else None
            )
            kernels[cid] = kernel_names(
                os.path.join(args.out_dir, "recheck", "logs"), cid
            )
    doc.update(candidates=candidates, kernels=kernels)
    _write_json(os.path.join(args.out_dir, "nightly.json"), doc)
    need = any(candidates.values())
    counts = {}
    for r in detect.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    text = summarize_exit(sched.meta, status) + (
        f"\nShard {args.shard}/{args.shards}: detection {json.dumps(counts)}; recheck of {len(recheck)} case(s): "
        f"{sum(v['verdict'] == 'regression' for v in verdicts.values())} regression(s), "
        f"{sum(v['verdict'] == 'improvement' for v in verdicts.values())} improvement(s).\n"
    )
    outputs = {
        "prev_xpu_ops_sha": args.prev_sha or "",
        "pytorch_sha": doc["cur"]["pytorch_sha"] or "",
    }
    if need:
        outputs["need_attribution"] = "true"
    return EXIT_OK, text, outputs


def attribution_cases(nightly_dir):
    """Cases with confirmed regressions and candidate commits, from all shards."""
    ids = set()
    for root, _, files in os.walk(nightly_dir):
        if "nightly.json" in files:
            doc = load_json(os.path.join(root, "nightly.json"), {})
            ids |= {c for c, cands in (doc.get("candidates") or {}).items() if cands}
    return sorted(ids)


def run_attribution(args, sel, cfg):
    ids = attribution_cases(args.nightly_dir)
    sides = {"P": args.python_prev, "Pp": args.python_base, "N": args.python}
    side_info = {
        "P": {"xpu_ops_sha": args.prev_sha, "role": "nightly_xpu_ops_vs_pytorch"},
        "Pp": {"xpu_ops_sha": args.prev_sha, "role": "nightly_xpu_ops_vs_pytorch"},
        "N": {"xpu_ops_sha": args.head_sha, "role": "nightly_xpu_ops_vs_pytorch"},
    }
    sched = Scheduler(args, sides, "multi")
    sched.meta["selected"] = len(ids)
    status = _measure(sched, ids, ["Pp", "N"], lambda s: s.measure_multi(ids, 6))
    status = sched.finish(status, side_info)
    _write_json(
        os.path.join(args.out_dir, "ab_cases.json"),
        {"scope": "nightly-xpu-ops-vs-pytorch", "cases": ids},
    )
    return status, summarize_exit(sched.meta, status), {}


RUNNERS = {
    "pr-vs-base": run_pr_vs_base,
    "nightly-vs-reference": run_nightly_vs_reference,
    "nightly-xpu-ops-vs-pytorch": run_attribution,
    "ondemand": run_ondemand,
}


def run_ci(args):
    os.makedirs(args.out_dir, exist_ok=True)
    sel = load_json(args.selection, {}) or {}
    cfg = Config(args.config_dir)
    t0 = time.time()
    status, text, outputs = RUNNERS[args.scope](args, sel, cfg)
    if sel:
        _write_json(os.path.join(args.out_dir, "selection.json"), sel)
    text = f"## microbench-run ({args.scope})\n\n{text}\nElapsed {time.time() - t0:.0f}s.\n"
    print(text)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    if args.github_output:
        with open(args.github_output, "a") as f:
            for k, v in outputs.items():
                f.write(f"{k}={v}\n")
    return status


def add_ci_parser(sub):
    p = sub.add_parser("ci-run", help="the microbench-run CI job of one scope")
    p.add_argument("--scope", required=True, choices=SCOPES)
    p.add_argument(
        "--selection", default=None, help="selection.json of microbench-select"
    )
    p.add_argument("--python", default=sys.executable, help="B / nightly (N) wheel")
    p.add_argument(
        "--python-base",
        default=None,
        help="base wheel (pr-vs-base), or P' (attribution)",
    )
    p.add_argument("--python-prev", default=None, help="previous nightly wheel P")
    p.add_argument("--state", default=None, help="nightly: reference.jsonl")
    p.add_argument("--state-run", default=None, help="nightly: run id of the state")
    p.add_argument(
        "--nightly-dir", default=None, help="attribution: shard results (nightly.json)"
    )
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--repo", default="..", help="git checkout for candidate commits")
    p.add_argument("--head-sha", default=None, help="xpu-ops commit of the B / N wheel")
    p.add_argument("--base-sha", default=None, help="xpu-ops commit of the base wheel")
    p.add_argument("--prev-sha", default=None, help="xpu-ops commit of P")
    p.add_argument("--prev-run-id", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--config-dir", default=None)
    p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    add_scheduler_args(p)
    p.add_argument(
        "--recheck-budget-min",
        type=float,
        default=120.0,
        help="nightly: time limit of the recheck",
    )
    p.set_defaults(func=lambda a: sys.exit(run_ci(a)))
