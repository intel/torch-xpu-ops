#!/usr/bin/env python3
"""Run upstream PyTorch unit tests on XPU with pytest, one file after another.

Each test file runs with the caller's PYTEST_ADDOPTS (in CI: timeout, xdist
workers one per GPU, ... from detect-devices.sh), except inductor's serial runs.

inductor
    The same tests as
      part1: run_test.py --inductor --include <INDUCTOR_PART1>
      part2: run_test.py --include <INDUCTOR_PART2>
    (--inductor only sets PYTORCH_TEST_WITH_INDUCTOR=1), run from pytorch/test
    with run_test.py's pytest options. Each file runs twice: `-m serial` in one
    process on the first GPU with PYTEST_ADDOPTS cleared, then `-m "not serial"`.

distributed
    Files under test/distributed/ from --files or the tracking issue, with XCCL.

default
    The other files from --files or the tracking issue. The issue list is the
    "Done test files" minus "Not Applicable test files" sections of
    intel/torch-xpu-ops#5205 (see --repo/--issue).

Examples (from the directory that contains pytorch/):
  run_upstream_ut.py default --list
  run_upstream_ut.py default --files test/test_nn.py test/test_ops.py
  run_upstream_ut.py inductor --gpus 0,1
  run_upstream_ut.py inductor --dry-run

Exit code: 0 all passed, 1 some tests failed, 2 setup error.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- common

ISSUE_API = "https://api.github.com/repos/{repo}/issues/{number}"
DISTRIBUTED_PREFIX = "test/distributed/"

COMMON_ENV = {
    "PYTORCH_TEST_WITH_SLOW": "1",
    "PYTORCH_TESTING_DEVICE_ONLY_FOR": "xpu",
}


def log(msg):
    print(msg, flush=True)


def elapsed(start):
    seconds = int(time.monotonic() - start)
    return f"{seconds // 60}m{seconds % 60:02d}s"


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


def get_issue_files(args, category):
    req = urllib.request.Request(
        ISSUE_API.format(repo=args.repo, number=args.issue),
        headers={"Accept": "application/vnd.github+json"},
    )
    if token := os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.load(resp).get("body") or ""

    done = parse_files(body, "Done test files")
    if not done:
        raise RuntimeError(f"No Done test files found in {args.repo}#{args.issue}")
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


@dataclass(frozen=True)
class Job:
    name: str  # log/xml file stem
    label: str
    cmd: list
    env: dict
    cwd: Path | None = None
    allow_empty: bool = False  # pytest exit code 5: nothing selected


def file_jobs(args, env, files):
    """One plain pytest run per file, from the current directory."""
    jobs = []
    for test_file in files:
        name = test_file.replace("/", "_")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            str(args.pytorch_dir / test_file),
            f"--junit-xml={args.xml_dir / f'{args.ut_name}_{name}.xml'}",
        ]
        jobs.append(Job(name, test_file, cmd, env))
    return jobs


def run_jobs(args, jobs):
    if args.dry_run:
        for job in jobs:
            cwd = f"(cd {job.cwd}) " if job.cwd else ""
            print(f"{cwd}{shlex.join(job.cmd)}")
        return 0

    start_all = time.monotonic()
    failed = 0
    for i, job in enumerate(jobs, 1):
        err_log = args.log_dir / f"{args.ut_name}_test_error_{job.name}.log"
        start = time.monotonic()
        print(f"::group::{job.label}", flush=True)
        with (
            open(args.log_dir / f"{args.ut_name}_test_{job.name}.log", "w") as out,
            open(err_log, "w") as err,
        ):
            # Tee stdout to the console and the log
            proc = subprocess.Popen(
                job.cmd,
                stdout=subprocess.PIPE,
                stderr=err,
                env=job.env,
                cwd=job.cwd,
                text=True,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                out.write(line)
            rc = proc.wait()
            if rc == 5 and job.allow_empty:
                rc = 0
            if rc != 0:
                err.write(job.label + "\n")
        print("::endgroup::", flush=True)
        failed += rc != 0
        log(
            f"[{i}/{len(jobs)}] [{'PASS' if rc == 0 else 'FAIL'}] {job.label} ({elapsed(start)})"
        )
    log(
        f"[SUMMARY] {args.ut_name}: {len(jobs)} runs, {len(jobs) - failed} passed, "
        f"{failed} failed in {elapsed(start_all)}"
    )
    return 1 if failed else 0


# ---------------------------------------------------------------- inductor

# Eager tests run through inductor
INDUCTOR_PART1 = [
    "test/test_modules.py",
    "test/test_ops.py",
    "test/test_ops_gradients.py",
    "test/test_torch.py",
]

# Inductor's own tests
INDUCTOR_PART2 = [
    "test/inductor/test_torchinductor.py",
    "test/inductor/test_torchinductor_opinfo.py",
    "test/inductor/test_aot_inductor.py",
    "test/inductor/test_cpu_select_algorithm.py",
]

INDUCTOR_ENV = {
    "BUILD_ENVIRONMENT": "linux-noble-xpu-n-py3.10-client",
    "TEST_CONFIG": "inductor_xpu",
    # Empty CI makes IS_CI false in common_utils
    "CI": "",
}

# run_test.py's pytest options, minus -x: its stop-and-retry loop isn't used here
INDUCTOR_PYTEST_ARGS = ["-vv", "-rfEX", "--reruns=2"]

# Serial runs ignore PYTEST_ADDOPTS (it may hold xdist options), so set the timeout here
SERIAL_PYTEST_ARGS = ["-p", "no:xdist", "--timeout=600", "--timeout_method=thread"]


def inductor_files(args):
    return INDUCTOR_PART1 + INDUCTOR_PART2


def inductor_jobs(args, env, files):
    env = {**env, **INDUCTOR_ENV}
    # Run from test/ like run_test.py does
    test_dir = (args.pytorch_dir / "test").resolve()
    xml_dir = args.xml_dir.resolve()
    jobs = []
    for test_file in files:
        part = 1 if test_file in INDUCTOR_PART1 else 2
        rel = test_file.removeprefix("test/")
        file_env = {**env, "PYTORCH_TEST_WITH_INDUCTOR": "1"} if part == 1 else env
        for marker in ("serial", "not serial"):
            name = f"part{part}_{rel.replace('/', '_')}"
            label = f"part{part} {test_file} -m '{marker}'"
            job_env = file_env
            extra = []
            if marker == "serial":
                name += "_serial"
                label += f" on GPU {args.gpus[0]}"
                job_env = {**file_env, "PYTEST_ADDOPTS": "", "ZE_AFFINITY_MASK": args.gpus[0]}
                extra = SERIAL_PYTEST_ARGS
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                rel,
                "-m",
                marker,
                *INDUCTOR_PYTEST_ARGS,
                *extra,
                f"--junit-xml-reruns={xml_dir / f'{args.ut_name}_{name}.xml'}",
            ]
            jobs.append(Job(name, label, cmd, job_env, test_dir, allow_empty=True))
    return jobs


# ---------------------------------------------------------------- distributed

DISTRIBUTED_ENV = {
    "BACKEND": "xccl",
    "WORLD_SIZE": "8",
    "USE_CCL_V2": "1",
}


def distributed_files(args):
    return args.files or get_issue_files(args, "distributed")


def distributed_jobs(args, env, files):
    env = {**env, **DISTRIBUTED_ENV, "ZE_AFFINITY_MASK": ",".join(args.gpus)}
    pipelining = (args.pytorch_dir / "test" / "distributed" / "pipelining").resolve()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (env.get("PYTHONPATH"), str(pipelining)) if p
    )
    if venv := env.get("VENV_ROOT"):
        env["PATH"] = f"{venv}/bin/libfabric{os.pathsep}{env['PATH']}"
    if not args.dry_run:
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
    return file_jobs(args, env, files)


# ---------------------------------------------------------------- default


def default_files(args):
    return args.files or get_issue_files(args, "default")


def default_jobs(args, env, files):
    return file_jobs(args, {**env, "ZE_AFFINITY_MASK": ",".join(args.gpus)}, files)


# ---------------------------------------------------------------- main

CATEGORIES = {
    "inductor": (inductor_files, inductor_jobs),
    "distributed": (distributed_files, distributed_jobs),
    "default": (default_files, default_jobs),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("category", choices=list(CATEGORIES))
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
        help="junit xml dir",
    )
    parser.add_argument(
        "--gpus",
        help="comma-separated GPU ids (default: $ZE_AFFINITY_MASK or 0; distributed: 0,1,2,3,4,5,6,7)",
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
        "0,1,2,3,4,5,6,7"
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
    get_files, get_jobs = CATEGORIES[args.category]
    try:
        files = get_files(args)
        if args.list:
            print("\n".join(files))
            return 0
        args.log_dir.mkdir(parents=True, exist_ok=True)
        args.xml_dir.mkdir(parents=True, exist_ok=True)
        jobs = get_jobs(args, {**os.environ, **COMMON_ENV}, files)
        return run_jobs(args, jobs)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
