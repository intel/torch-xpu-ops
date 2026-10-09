#!/usr/bin/env python3
"""Run upstream PyTorch unit tests on XPU, in one of two modes.

run_test.py shards (category: inductor)
    Each file in INDUCTOR_PARTS is split into --shards-per-file shards via
    `run_test.py --shard`. Shards are queued in order and each GPU runs one
    shard at a time, taking the next one as soon as it is free.

pytest per file (categories: default, distributed)
    One pytest run per test file, sequentially. Files come from --files or
    from the "Done test files" minus "Not Applicable test files" sections of
    the tracking issue (default: intel/torch-xpu-ops#5205). distributed takes
    files under test/distributed/ and sets up XCCL; default takes the rest.

Examples (from the directory that contains pytorch/):
  run_upstream_ut.py default --list
  run_upstream_ut.py default --files test/test_nn.py test/test_ops.py
  run_upstream_ut.py inductor --gpus 0,1 --shards-per-file 2
  run_upstream_ut.py inductor --dry-run

Exit code: 0 all passed, 1 some tests failed, 2 setup error.
"""

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ISSUE_API = "https://api.github.com/repos/{repo}/issues/{number}"
DISTRIBUTED_PREFIX = "test/distributed/"
PASS_LOG_TAIL = 200  # lines of a passing inductor shard's log echoed to the console

# part -> (extra run_test.py flags, test files)
INDUCTOR_PARTS = {
    # Eager tests run through inductor
    "part1": (
        ["--inductor"],
        ["test_modules", "test_ops", "test_ops_gradients", "test_torch"],
    ),
    # Inductor's own tests; no --inductor to avoid nested dynamo state
    "part2": (
        [],
        [
            "inductor/test_torchinductor",
            "inductor/test_torchinductor_opinfo",
            "inductor/test_aot_inductor",
            "inductor/test_cpu_select_algorithm",
        ],
    ),
}

COMMON_ENV = {
    "PYTORCH_TEST_WITH_SLOW": "1",
    "PYTORCH_TESTING_DEVICE_ONLY_FOR": "xpu",
}

INDUCTOR_ENV = {
    "BUILD_ENVIRONMENT": "linux-noble-xpu-n-py3.10-client",
    "TEST_CONFIG": "inductor_xpu",
    "PYTORCH_RETRY_TEST_CASES": "1",
    "PYTORCH_OVERRIDE_FLAKY_SIGNAL": "1",
    "CONTINUE_THROUGH_ERROR": "True",
    "PYTORCH_TEST_RERUN_DISABLED_TESTS": "0",
    "NO_TEST_TIMEOUT": "False",
    "VERBOSE_TEST_LOGS": "False",
    "TEST_SHOWLOCALS": "False",
    "NO_TD": "False",
    # Empty CI makes IS_CI false, disabling run_test.py's S3/TD/metrics/upload paths
    "CI": "",
    # One test process per shard: NUM_PROCS is 3 on XPU and, unlike ROCm, is never
    # clamped to the GPU count, so the default would put 3 workers on the one GPU
    # this shard is pinned to via ZE_AFFINITY_MASK.
    "PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL": "1",
    "PYTEST_ADDOPTS": " --timeout 3600 --timeout_method=thread -v ",
}

DISTRIBUTED_ENV = {
    "BACKEND": "xccl",
    "WORLD_SIZE": "4",
    "USE_CCL_V2": "1",
}

_print_lock = threading.Lock()


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def elapsed(start):
    seconds = int(time.monotonic() - start)
    return f"{seconds // 60}m{seconds % 60:02d}s"


# ---------------------------------------------------------------- issue list


def parse_files(body, heading):
    """Backticked *.py paths in the <details> block that starts at heading."""
    body = body[max(body.find("auto-file-lists:begin"), 0) :]
    start = body.find(heading)
    if start == -1:
        return []
    end = body.find("</details>", start)
    section = body[start : end if end != -1 else None]
    # Paths may be wrapped across lines inside the backticks
    paths = (re.sub(r"\s+", "", m) for m in re.findall(r"`([^`]+?\.py)`", section))
    return list(dict.fromkeys(p for p in paths if p))


def get_issue_files(repo, issue, category):
    req = urllib.request.Request(
        ISSUE_API.format(repo=repo, number=issue),
        headers={"Accept": "application/vnd.github+json"},
    )
    if token := os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp).get("body") or ""

    done = parse_files(body, "Done test files")
    if not done:
        raise RuntimeError(f"No Done test files found in {repo}#{issue}")
    excluded = set(parse_files(body, "Not Applicable test files"))
    files = [f for f in done if f not in excluded]
    distributed = [f for f in files if f.startswith(DISTRIBUTED_PREFIX)]
    print(
        f"Done test files: {len(files)} (distributed: {len(distributed)}, "
        f"default: {len(files) - len(distributed)}; "
        f"excluded {len(done) - len(files)} Not Applicable)",
        file=sys.stderr,
    )
    if category == "distributed":
        return distributed
    return [f for f in files if not f.startswith(DISTRIBUTED_PREFIX)]


# ---------------------------------------------------------------- run_test.py shards


@dataclass(frozen=True)
class Shard:
    id: int  # global, 1..N in queue order
    part: str
    test: str
    index: int  # 1..total within this test file
    total: int

    @property
    def tag(self):
        return f"shard{self.id}_{self.part}_{self.test.replace('/', '_')}_{self.index}of{self.total}"

    @property
    def args(self):
        extra, _ = INDUCTOR_PARTS[self.part]
        return [
            *extra,
            "--include",
            self.test,
            "--shard",
            str(self.index),
            str(self.total),
        ]


def build_shards(per_file):
    tests = [(part, t) for part, (_, ts) in INDUCTOR_PARTS.items() for t in ts]
    return [
        Shard(n * per_file + i, part, test, i, per_file)
        for n, (part, test) in enumerate(tests)
        for i in range(1, per_file + 1)
    ]


def run_shard(shard, gpu, args, env, out_log, err_log):
    xml_dir = args.log_dir / "xml" / shard.tag
    cmd = [
        sys.executable,
        str(args.pytorch_dir / "test" / "run_test.py"),
        "--verbose",
        *shard.args,
        "--save-xml",
        str(xml_dir),
    ]
    with open(out_log, "w") as out, open(err_log, "w") as err:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=err,
            env={**env, "ZE_AFFINITY_MASK": gpu},
            start_new_session=True,
        )
        rc = proc.wait()
    # Kill leftover children so the GPU is free before the next shard starts
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return rc


def run_shards(args, env):
    shards = build_shards(args.shards_per_file)
    if args.dry_run:
        for s in shards:
            print(f"shard{s.id}: {s.part} {' '.join(s.args).replace('--include ', '')}")
        return 0

    shutil.rmtree(args.log_dir / "xml", ignore_errors=True)
    log(
        f"[INFO] inductor: {len(shards)} shards on {len(args.gpus)} GPUs ({' '.join(args.gpus)})"
    )
    free_gpus = queue.SimpleQueue()
    for gpu in args.gpus:
        free_gpus.put(gpu)
    failed = []
    finished = 0

    def task(shard):
        nonlocal finished
        gpu = free_gpus.get()  # never blocks: one pool worker per GPU
        start = time.monotonic()
        out_log = args.log_dir / f"{args.ut_name}_test_{shard.tag}.log"
        err_log = args.log_dir / f"{args.ut_name}_test_error_{shard.tag}.log"
        log(f"[START] gpu {gpu}: {shard.tag}")
        try:
            rc = run_shard(shard, gpu, args, env, out_log, err_log)
        except Exception as e:  # noqa: BLE001 - count as failed, keep going
            log(f"[ERROR] gpu {gpu}: {shard.tag}: {e}")
            rc = 255
        finally:
            free_gpus.put(gpu)
        # Dump the finished shard's logs in one block so parallel shards don't interleave.
        # Passing shards only show the tail to stay under the CI log size limit.
        with _print_lock:
            finished += 1
            if rc != 0:
                failed.append(shard.tag)
            result = (
                f"[{'PASS' if rc == 0 else 'FAIL'}] [{finished}/{len(shards)}] "
                f"run_test.py {' '.join(shard.args)} ({elapsed(start)})"
            )
            print(f"::group::{result}", flush=True)
            for path in (out_log, err_log):
                if not path.exists():
                    continue
                with open(path, errors="replace") as f:
                    if rc != 0:
                        shutil.copyfileobj(f, sys.stdout)
                        continue
                    tail = deque(enumerate(f, 1), maxlen=PASS_LOG_TAIL)
                    if tail and tail[0][0] > 1:
                        print(f"... last {len(tail)} lines, full log: {path}")
                    sys.stdout.writelines(line for _, line in tail)
            print("::endgroup::", flush=True)
            print(result, flush=True)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(task, shards))
    log(f"[TIME] inductor all ({elapsed(start)})")
    if not failed:
        log(f"[SUMMARY] {args.ut_name}: all {len(shards)} shards passed")
        return 0
    log(
        f"[SUMMARY] {args.ut_name}: {len(failed)}/{len(shards)} shards failed: {' '.join(sorted(failed))}"
    )
    return 1


# ---------------------------------------------------------------- pytest per file


def setup_distributed_env(env, pytorch_dir):
    env.update(DISTRIBUTED_ENV)
    pipelining = (pytorch_dir / "test" / "distributed" / "pipelining").resolve()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (env.get("PYTHONPATH"), str(pipelining)) if p
    )
    if venv := env.get("VENV_ROOT"):
        env["PATH"] = f"{venv}/bin/libfabric{os.pathsep}{env['PATH']}"
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch;print(torch.distributed.is_xccl_available())",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if out.lower() in ("false", "0"):
        raise RuntimeError("XCCL is not enabled")


def run_files(args, env, files):
    if args.dry_run:
        print("\n".join(files))
        return 0

    start_all = time.monotonic()
    failed = 0
    for i, test_file in enumerate(files, 1):
        name = test_file.replace("/", "_")
        err_log = args.log_dir / f"{args.ut_name}_test_error_{name}.log"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            str(args.pytorch_dir / test_file),
            f"--junit-xml={args.xml_dir / f'{args.ut_name}_{name}.xml'}",
        ]
        start = time.monotonic()
        print(f"::group::{test_file}", flush=True)
        with (
            open(args.log_dir / f"{args.ut_name}_test_{name}.log", "w") as out,
            open(err_log, "w") as err,
        ):
            # Tee stdout to the console and the log
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=err, env=env, text=True
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                out.write(line)
            rc = proc.wait()
            if rc != 0:
                err.write(test_file + "\n")
        print("::endgroup::", flush=True)
        failed += rc != 0
        log(
            f"[{i}/{len(files)}] [{'PASS' if rc == 0 else 'FAIL'}] {test_file} ({elapsed(start)})"
        )
    log(
        f"[SUMMARY] {args.ut_name}: {len(files)} files, {len(files) - failed} passed, "
        f"{failed} failed in {elapsed(start_all)}"
    )
    return 1 if failed else 0


# ---------------------------------------------------------------- main


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("category", choices=["inductor", "default", "distributed"])
    parser.add_argument(
        "--ut-name", help="log/xml name prefix (default: upstream_<category>)"
    )
    parser.add_argument(
        "--pytorch-dir",
        type=Path,
        default=Path("pytorch"),
        help="PyTorch source checkout",
    )
    parser.add_argument("--log-dir", type=Path, help="default: ut_log/<ut-name>")
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("ut_log"),
        help="junit xml dir for default/distributed",
    )
    parser.add_argument(
        "--gpus",
        help="comma-separated GPU ids (default: $ZE_AFFINITY_MASK or 0; distributed: 0,1,2,3)",
    )
    parser.add_argument(
        "--shards-per-file",
        type=int,
        default=2,
        help="inductor: run_test.py shards per test file",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="default/distributed: test files instead of the issue list",
    )
    parser.add_argument(
        "--repo", default="intel/torch-xpu-ops", help="owner/repo of the tracking issue"
    )
    parser.add_argument("--issue", type=int, default=5205, help="tracking issue number")
    parser.add_argument(
        "--list", action="store_true", help="print the test file list and exit"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would run and exit"
    )
    args = parser.parse_args()

    args.ut_name = args.ut_name or f"upstream_{args.category}"
    args.log_dir = args.log_dir or Path("ut_log") / args.ut_name
    default_gpus = (
        "0,1,2,3"
        if args.category == "distributed"
        else os.environ.get("ZE_AFFINITY_MASK") or "0"
    )
    # Dedupe so each GPU gets exactly one worker
    args.gpus = list(
        dict.fromkeys(
            g.strip() for g in (args.gpus or default_gpus).split(",") if g.strip()
        )
    )
    return args


def main():
    args = parse_args()
    try:
        if args.category == "inductor":
            if args.list:
                for part, (extra, tests) in INDUCTOR_PARTS.items():
                    print(f"{part}: {' '.join(extra + tests)}")
                return 0
        else:
            files = args.files or get_issue_files(args.repo, args.issue, args.category)
            if args.list:
                print("\n".join(files))
                return 0

        args.log_dir.mkdir(parents=True, exist_ok=True)
        args.xml_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **COMMON_ENV}
        if args.category == "inductor":
            return run_shards(args, {**env, **INDUCTOR_ENV})
        env["ZE_AFFINITY_MASK"] = ",".join(args.gpus)
        if args.category == "distributed" and not args.dry_run:
            setup_distributed_env(env, args.pytorch_dir)
        return run_files(args, env, files)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
