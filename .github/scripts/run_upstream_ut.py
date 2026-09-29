#!/usr/bin/env python3
"""Run upstream PyTorch unit tests on XPU.

Categories:
  inductor     Fixed inductor test set, run via pytorch/test/run_test.py.
               Each test file is split into --shards-per-file shards, all put
               into one queue; each GPU runs one shard at a time and picks
               the next one as soon as it is free.
  default      "Done" non-distributed files from the tracking issue, one
               pytest run per file.
  distributed  "Done" test/distributed/ files from the tracking issue, one
               pytest run per file, with the XCCL environment set up.

The file list for default/distributed comes from the tracking issue
(default: intel/torch-xpu-ops#5205), between the
``<!-- auto-file-lists:begin -->`` / ``<!-- auto-file-lists:end -->`` markers:
files under "Done test files" minus those under "Not Applicable test files".

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
from typing import NamedTuple

API_URL = "https://api.github.com/repos/{owner}/{repo}/issues/{number}"
DISTRIBUTED_PREFIX = "test/distributed/"

# name -> (extra run_test.py flags, --include tests)
INDUCTOR_PARTS = {
    # Eager to inductor cases
    "part1": (
        ["--inductor"],
        ["test_modules", "test_ops", "test_ops_gradients", "test_torch"],
    ),
    # Inductor own tests; no --inductor to avoid nested dynamo state
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


def fmt_elapsed(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s"


# ---------------------------------------------------------------- file list


def fetch_issue_body(repo, number):
    owner, _, name = repo.partition("/")
    req = urllib.request.Request(API_URL.format(owner=owner, repo=name, number=number))
    req.add_header("Accept", "application/vnd.github+json")
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp).get("body") or ""


def extract_section(body, heading):
    begin = body.find("auto-file-lists:begin")
    if begin != -1:
        body = body[begin:]
    start = body.find(heading)
    if start == -1:
        return ""
    end = body.find("</details>", start)
    return body[start:] if end == -1 else body[start:end]


def parse_files(section):
    files = []
    for match in re.findall(r"`([^`]+?\.py)`", section):
        path = re.sub(r"\s+", "", match)
        if path and path not in files:
            files.append(path)
    return files


def get_issue_files(repo, issue, category):
    body = fetch_issue_body(repo, issue)
    done = parse_files(extract_section(body, "Done test files"))
    if not done:
        raise RuntimeError(f"No Done test files found in {repo}#{issue}")
    not_applicable = set(
        parse_files(extract_section(body, "Not Applicable test files"))
    )
    files = [f for f in done if f not in not_applicable]
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


# ---------------------------------------------------------------- inductor


class Job(NamedTuple):
    shard: int  # global id, 1..len(jobs)
    part: str
    test: str  # single run_test.py --include target
    file_shard: int  # 1..shards_per_file within this test file
    shards_per_file: int

    @property
    def tag(self):
        return f"shard{self.shard}_{self.part}_{self.test.replace('/', '_')}_{self.file_shard}of{self.shards_per_file}"


def build_jobs(shards_per_file):
    """Split every test file into shards_per_file shards, numbered globally in order."""
    jobs = []
    for part, (_, tests) in INDUCTOR_PARTS.items():
        for test in tests:
            for i in range(1, shards_per_file + 1):
                jobs.append(Job(len(jobs) + 1, part, test, i, shards_per_file))
    return jobs


def run_shard(job, gpu, args, env):
    """Run one shard in its own process group; return its exit code."""
    extra, _ = INDUCTOR_PARTS[job.part]
    xml_dir = os.path.join(args.log_dir, "xml", job.tag)
    shutil.rmtree(xml_dir, ignore_errors=True)
    cmd = [
        sys.executable,
        os.path.join(args.pytorch_dir, "test", "run_test.py"),
        "--verbose",
        *extra,
        "--include",
        job.test,
        "--shard",
        str(job.file_shard),
        str(job.shards_per_file),
        "--save-xml",
        xml_dir,
    ]
    with (
        open(
            os.path.join(args.log_dir, f"{args.ut_name}_test_{job.tag}.log"), "w"
        ) as out,
        open(
            os.path.join(args.log_dir, f"{args.ut_name}_test_error_{job.tag}.log"), "w"
        ) as err,
    ):
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=err,
            env={**env, "ZE_AFFINITY_MASK": gpu},
            start_new_session=True,
        )
        rc = proc.wait()
    # Reap leftover children so the GPU is free before the next shard starts
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return rc


def gpu_worker(gpu, jobs, results, args, env):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        start = time.time()
        log(f"[START] gpu {gpu}: {job.tag}")
        try:
            rc = run_shard(job, gpu, args, env)
        except Exception as e:  # noqa: BLE001 - record and keep draining the queue
            log(f"[ERROR] gpu {gpu}: {job.tag}: {e}")
            rc = 255
        results[job.tag] = rc
        log(f"[DONE] gpu {gpu}: {job.tag} rc={rc} ({fmt_elapsed(time.time() - start)})")


def run_inductor(args, env):
    all_jobs = build_jobs(args.shards_per_file)
    if args.dry_run:
        for j in all_jobs:
            extra, _ = INDUCTOR_PARTS[j.part]
            print(
                f"shard{j.shard}: {j.part} {' '.join(extra + [j.test])} --shard {j.file_shard} {j.shards_per_file}"
            )
        return 0
    shutil.rmtree(os.path.join(args.log_dir, "xml"), ignore_errors=True)
    log(
        f"[INFO] inductor: {len(all_jobs)} shards on {len(args.gpus)} GPUs ({' '.join(args.gpus)})"
    )
    start = time.time()
    results = {}
    pending = all_jobs
    # A shard only lacks a result if its worker thread died; requeue those
    for round_id in range(1, 4):
        jobs = queue.Queue()
        for job in pending:
            jobs.put(job)
        threads = [
            threading.Thread(target=gpu_worker, args=(gpu, jobs, results, args, env))
            for gpu in args.gpus
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pending = [j for j in all_jobs if j.tag not in results]
        if not pending:
            break
        log(f"[RETRY] round {round_id}: requeue {len(pending)} shards without result")
    log(f"[TIME] inductor all ({fmt_elapsed(time.time() - start)})")
    failed = sorted(tag for tag, rc in results.items() if rc != 0)
    missing = len(all_jobs) - len(results)
    if not failed and not missing:
        log(f"[SUMMARY] {args.ut_name}: all {len(all_jobs)} shards passed")
        return 0
    log(
        f"[SUMMARY] {args.ut_name}: {len(results)}/{len(all_jobs)} shards finished, "
        f"failed: {' '.join(failed)}"
    )
    return 1


# ---------------------------------------------------------------- per-file


def run_files(args, env, files):
    if args.dry_run:
        print("\n".join(files))
        return 0
    start_all = time.time()
    passed = failed = 0
    for test_file in files:
        log_name = test_file.replace("/", "_")
        err_log = os.path.join(
            args.log_dir, f"{args.ut_name}_test_error_{log_name}.log"
        )
        out_log = os.path.join(args.log_dir, f"{args.ut_name}_test_{log_name}.log")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            os.path.join(args.pytorch_dir, test_file),
            f"--junit-xml={os.path.join(args.xml_dir, f'{args.ut_name}_{log_name}.xml')}",
        ]
        start = time.time()
        print(f"::group::{test_file}", flush=True)
        with open(out_log, "w") as out, open(err_log, "w") as err:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=err, env=env, text=True
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                out.write(line)
            rc = proc.wait()
        print("::endgroup::", flush=True)
        if rc == 0:
            status, passed = "PASS", passed + 1
        else:
            status, failed = "FAIL", failed + 1
            with open(err_log, "a") as err:
                err.write(test_file + "\n")
        log(f"[{status}] {test_file} ({fmt_elapsed(time.time() - start)})")
    log(
        f"[SUMMARY] {args.ut_name}: {len(files)} files, {passed} passed, {failed} failed "
        f"in {fmt_elapsed(time.time() - start_all)}"
    )
    return 1 if failed else 0


def setup_distributed_env(env, pytorch_dir):
    env.update(DISTRIBUTED_ENV)
    pipelining = os.path.abspath(
        os.path.join(pytorch_dir, "test", "distributed", "pipelining")
    )
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (env.get("PYTHONPATH"), pipelining) if p
    )
    if env.get("VENV_ROOT"):
        env["PATH"] = f"{env['VENV_ROOT']}/bin/libfabric{os.pathsep}{env['PATH']}"
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


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("category", choices=["inductor", "default", "distributed"])
    parser.add_argument(
        "--ut-name", help="log/xml name prefix (default: upstream_<category>)"
    )
    parser.add_argument(
        "--pytorch-dir", default="pytorch", help="PyTorch source checkout"
    )
    parser.add_argument("--log-dir", help="default: ut_log/<ut-name>")
    parser.add_argument(
        "--xml-dir", default="ut_log", help="junit xml dir for default/distributed"
    )
    parser.add_argument(
        "--gpus",
        help="comma-separated GPU ids "
        "(default: $ZE_AFFINITY_MASK or 0; distributed: 0,1,2,3)",
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
    args.log_dir = args.log_dir or os.path.join("ut_log", args.ut_name)
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

    try:
        if args.category != "inductor":
            files = args.files or get_issue_files(args.repo, args.issue, args.category)
        if args.list:
            print(
                "\n".join(files)
                if args.category != "inductor"
                else "\n".join(
                    f"{p}: {' '.join(e + t)}" for p, (e, t) in INDUCTOR_PARTS.items()
                )
            )
            return 0
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.xml_dir, exist_ok=True)
        env = {**os.environ, **COMMON_ENV}
        if args.category == "inductor":
            env.update(INDUCTOR_ENV)
            return run_inductor(args, env)
        if args.category == "distributed":
            env["ZE_AFFINITY_MASK"] = ",".join(args.gpus)
            if not args.dry_run:
                setup_distributed_env(env, args.pytorch_dir)
        return run_files(args, env, files)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
