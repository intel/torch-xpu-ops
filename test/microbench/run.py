# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""A/B scheduler and machine environment (no torch in this process).

``python -m microbench schedule`` drives worker processes over the cards;
``python -m microbench env`` shows the layout, locks GPU frequency and runs
the pre-run checks.
"""

import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

from . import (
    CANARY_ID,
    CONFIG_DIR,
    DEFAULT_CASE_SECONDS,
    judge,
    load_history,
    load_json,
    load_yaml,
    PACKAGE_DIR,
    read_jsonl,
)

# ---------------------------------------------------------------------------
# Machine layout, GPU frequency, pre-run checks (plan 3.3, 3.4)
# ---------------------------------------------------------------------------
# Cards, NUMA nodes and CPU sets come from ``config/machines.yaml`` when the
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
    data = load_yaml(path or os.path.join(CONFIG_DIR, "machines.yaml"), {})
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
    p.add_argument("--machines", default=None, help="default: config/machines.yaml")
    p.add_argument("--ratio", type=float, default=0.95, help="lock at ratio * RP0")
    p.add_argument("--mhz", type=int, default=None, help="lock at this frequency")
    p.set_defaults(func=lambda a: sys.exit(run_env(a)))


# ---------------------------------------------------------------------------
# Scheduler (plan 3.3, 7.2)
# ---------------------------------------------------------------------------
# For every card, rounds alternate ``A B | B A | A B ...``; each (side, round)
# is a fresh worker process started with that side's venv python, pinned to
# the card (``ZE_AFFINITY_MASK``) and to the card's NUMA node / CPUs. Both
# sides run the same benchmark code (this checkout); only the wheel differs.
#
# Stages: ``validate`` (changed cases, side B), ``canary_start``, ``screen``
# (all cases), ``confirm`` (suspicious cases, B first, moved to the next card,
# A and B on the same card), ``canary_end``. Only raw data is recorded; the
# check job makes the verdicts.

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


def round_order(r, first="A"):
    other = "B" if first == "A" else "A"
    return [first, other] if r % 2 == 0 else [other, first]


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


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out_dir)
        self.raw = os.path.join(self.out, "raw")
        os.makedirs(self.raw, exist_ok=True)
        self.machine = machine_config(args.machines)
        cards = self.machine["cards"]
        if args.cards:
            want = [int(c) for c in args.cards.split(",")]
            cards = [c for c in cards if c["index"] in want] or [
                {"index": i} for i in want
            ]
        self.cards = cards or [{"index": 0}]
        self.python = {"A": args.python_a or args.python_b, "B": args.python_b}
        same = os.path.realpath(self.python["A"]) == os.path.realpath(self.python["B"])
        self.protocol = dict(self.machine.get("protocol", {}))
        self.protocol.update(load_json(args.protocol, {}))
        self.protocol_path = os.path.join(self.out, "protocol.json")
        with open(self.protocol_path, "w") as f:
            json.dump(self.protocol, f, indent=2)
        self.costs = load_history(args.history)["case_cost"]
        self.deadline = time.time() + args.budget_min * 60 if args.budget_min else None
        self.meta = {
            "mode": "aa" if same else "ab",
            "host": self.machine.get("host"),
            "cards": [
                {k: c.get(k) for k in ("index", "drm", "numa", "cpus")}
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
            argv += ["--quick"] if self.args.quick else []
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
                    mine, type="result", case_id=crashed, status="crashed", rc=str(rc)
                )
                f.write(json.dumps(rec) + "\n")
            remaining.remove(crashed)

    def run_stage(self, stage, shards, rounds, first="A", sides=("A", "B")):
        t0 = time.time()
        info = self.meta["stages"].setdefault(stage, {})

        def card_loop(card, ids):
            for r in range(rounds):
                if self.deadline and time.time() > self.deadline:
                    with self.lock:
                        info["truncated_at_round"] = r
                    return
                for side in round_order(r, first):
                    if side in sides:
                        self.run_worker(side, card, stage, r, ids)

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
            seconds=round(time.time() - t0, 1),
            rounds=rounds,
            first=first,
            shards={str(c["index"]): ids for c, ids in zip(self.cards, shards)},
        )

    def check_versions(self):
        shas = {side: torch_git_version(py) for side, py in self.python.items()}
        self.meta["torch_git_version"] = shas
        if shas["A"] != shas["B"] and not self.args.allow_sha_mismatch:
            raise InfraError(
                f"PyTorch SHA differs between wheels: A={shas['A']} B={shas['B']}"
            )
        want = self.args.expect_torch_sha
        if want and not (
            shas["B"].startswith(want[:12]) or want.startswith(shas["B"][:12])
        ):
            raise InfraError(f"PyTorch SHA {shas['B']} != resolved SHA {want}")

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
        out = os.path.join(self.out, "validation.jsonl")
        case_file = os.path.join(self.raw, "validate.cases.json")
        with open(case_file, "w") as f:
            json.dump(ids, f)
        budget = self.args.validate_budget_min
        cmd = [
            "validate",
            "--cases",
            case_file,
            "--out",
            out,
            "--budget-min",
            str(budget),
        ]
        try:
            # the budget is checked between cases; allow one slow case on top
            timeout = budget * 60 + 600 if budget else None
            proc = self._spawn(
                "B", self.cards[0], cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            raise InfraError(f"validator did not finish within {timeout}s") from e
        sys.stderr.write(proc.stderr)
        results = read_jsonl(out)
        failed = [r for r in results if r.get("status") == "fail"]
        keys = ("case_id", "op", "pass_", "label", "failures")
        self.meta["validation"] = {
            "cases": len(ids),
            "failed": [{k: r.get(k) for k in keys} for r in failed],
            "not_validated": [
                r["case_id"] for r in results if r.get("status") == "not_validated"
            ],
        }
        if len(results) < len(ids):
            raise InfraError(
                f"validator produced {len(results)}/{len(ids)} results (rc={proc.returncode})"
            )
        return failed

    def suspicious(self, ids):
        recs = []
        for name in os.listdir(self.raw):
            if name.startswith("screen_") and name.endswith(".jsonl"):
                recs += read_jsonl(os.path.join(self.raw, name))
        groups = judge.group_samples(
            recs, "e2e_us" if self.args.device == "cpu" else "device_us"
        )
        sus = []
        for cid in ids:
            g = groups.get((cid, "screen"))
            if g and g["A"] and g["B"]:
                res = judge.compare(g["A"], g["B"], n_boot=500)
                if judge.suspicious(res) or judge.classify(res)[0] in (
                    "regression",
                    "improvement",
                ):
                    sus.append(cid)
        return sus

    def run(self):
        a = self.args
        ids = _selection_list(a.cases, "cases")
        self.meta["selected"] = len(ids)
        status = EXIT_OK
        try:
            self.check_versions()
            if a.env_check:
                problems, self.meta["env_check"] = check_host(
                    [c for c in self.cards if c.get("drm")],
                    self.machine.get("limits", {}),
                )
                if problems:
                    raise InfraError("environment check failed: " + "; ".join(problems))
            self.snapshot_env("env_start")
            validate_ids = _selection_list(a.validate, "validate")
            if validate_ids and self.validate(validate_ids):
                return EXIT_VALIDATION
            if not ids:
                return status
            canary = [[CANARY_ID] for _ in self.cards]
            if a.canary:
                self.run_stage("canary_start", canary, 1, first="B", sides=("B",))
            shards, load = lpt_shards(ids, self.costs, len(self.cards))
            self.meta["estimated_load_s"] = [round(x, 1) for x in load]
            self.run_stage("screen", shards, a.screen_rounds)
            if a.confirm_rounds > 0 and self.deadline and time.time() > self.deadline:
                self.meta["confirm_skipped"] = "time budget exhausted"
            elif a.confirm_rounds > 0:
                sus = self.meta["suspicious"] = self.suspicious(ids)
                # each suspicious case moves to the next card (A and B together)
                n = len(self.cards)
                cshards = [[] for _ in range(n)]
                for k, shard in enumerate(shards):
                    cshards[(k + 1) % n] += [c for c in shard if c in sus]
                if sus:
                    self.run_stage("confirm", cshards, a.confirm_rounds, first="B")
            if a.canary:
                self.run_stage("canary_end", canary, 1, first="B", sides=("B",))
            self.snapshot_env("env_end")
            if self.meta["infra_errors"]:
                status = EXIT_INFRA
        except InfraError as e:
            self.meta["infra_errors"].append(str(e))
            status = EXIT_INFRA
        finally:
            self.meta.update(finished=time.time(), exit_status=status)
            with open(os.path.join(self.out, "run_meta.json"), "w") as f:
                json.dump(self.meta, f, indent=2)
        return status


def summarize_exit(meta, status):
    if status == EXIT_INFRA:
        lines = ["### Benchmark did not complete (infra error)", ""]
        lines += [f"- {e}" for e in meta.get("infra_errors", [])]
        lines += [
            "",
            "Please rerun the failed job. The override label cannot accept infrastructure errors.",
        ]
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


def run_schedule(args):
    sched = Scheduler(args)
    status = sched.run()
    text = summarize_exit(sched.meta, status)
    print(text)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(text)
    return status


def add_schedule_parser(sub):
    p = sub.add_parser("schedule", help="run A/B (or A/A) measurement across cards")
    p.add_argument(
        "--python-a", default=None, help="side A (base wheel); default: B (A/A)"
    )
    p.add_argument("--python-b", default=sys.executable, help="side B (PR wheel)")
    p.add_argument(
        "--cases", required=True, help="JSON list of case_ids or selection.json"
    )
    p.add_argument(
        "--validate", default=None, help="cases to validate first (same format)"
    )
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
    p.add_argument("--screen-rounds", type=int, default=3)
    p.add_argument("--confirm-rounds", type=int, default=6)
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
        "--allow-sha-mismatch", action="store_true", help="nightly: A and B differ"
    )
    p.add_argument(
        "--env-check",
        action="store_true",
        help="fail on busy card / unlocked freq / hot GPU",
    )
    p.add_argument("--canary", action="store_true")
    p.add_argument(
        "--quick", action="store_true", help="smoke mode: 3 iterations, no warmup"
    )
    p.add_argument(
        "--summary",
        default=None,
        help="append a markdown summary (GITHUB_STEP_SUMMARY)",
    )
    p.set_defaults(func=lambda a: sys.exit(run_schedule(a)))
