#!/usr/bin/env python3
"""
run_blank_test.py — Run unclassified test cases locally and record "Local Passed".

Gate 0 of classify_ut workflow:
1. Reads tasks.json (output of extract_tasks.py)
2. For each task with blank status_xpu AND blank Reason, runs pytest locally
3. If the test passes locally: Reason="Local Passed", DetailReason=torch version
4. If the test fails/skips/times out: left as-is for cascade
5. Dumps test logs to test_logs/ directory
6. Outputs results.json (Local Passed tasks + remaining tasks for cascade)

Per-case timeout: each pytest invocation is run with the `pytest-timeout` plugin
(`--timeout <SECONDS> --timeout_method=thread`, matching CI's own convention -
see AGENTS.md), so a single hanging test is interrupted by pytest itself
(thread-based, so it can interrupt C-extension calls a signal-based timeout
cannot) rather than relying solely on an external process kill. The outer
`subprocess.run(..., timeout=...)` watchdog is kept as a safety-net kill-switch
(SECONDS + a small buffer) in case pytest-timeout cannot intervene (e.g. a hang
during collection, before any test has started).

Requires the `pytest-timeout` plugin to be installed in the target environment
(installed by setup_env.sh; if missing, this script fails fast with a clear
message rather than installing it).

Results written to results.json can be fed into write_results.py for Excel output.

Usage:
    python3 run_blank_test.py <tasks.json> [--output results.json] [--log-dir test_logs]
                             [--timeout SECONDS] [--env CONDA_ENV_NAME]
                             [--pytorch-root DIR]

    --pytorch-root DIR  Directory that relative test files are resolved against
                        and that pytest runs in. Defaults to the PYTORCH_FOLDER
                        environment variable, or the current directory if unset.
    --timeout SECONDS   Per-test-case timeout enforced by pytest-timeout
                        (--timeout_method=thread). Default: 600 (matches CI).

Examples:
    python3 run_blank_test.py tasks.json --output results.json
    python3 run_blank_test.py tasks.json --output results.json --timeout 600
    python3 run_blank_test.py tasks.json --pytorch-root "$HOME/daisy_pytorch"
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Safety-net buffer added on top of the per-case pytest-timeout value when
# setting the outer subprocess.run() watchdog, so pytest-timeout's own
# thread-based timeout gets a chance to fire and report gracefully before the
# harder external kill (which loses any in-flight pytest-timeout traceback).
WATCHDOG_BUFFER_SECONDS = 60


def _isolated(base_cmd):
    """Prefix with -I so subprocess doesn't pick up local cwd packages."""
    return base_cmd + ["-I"]

def _isolated_cmd(base_cmd, code):
    return _isolated(base_cmd) + ["-c", code]

def get_torch_version(conda_env=None):
    base = [sys.executable]
    if conda_env:
        base = ["conda", "run", "-n", conda_env, "python3"]
    cmd = _isolated_cmd(base, "import torch; print(torch.__version__)")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def check_environment(conda_env=None):
    """Check torch environment before running tests. Returns (ok, info_dict)."""
    info = {"conda_env": conda_env, "torch_available": False, "torch_version": "unknown",
            "cuda_available": False, "xpu_available": False, "python_ok": False}

    if conda_env:
        # Check conda env exists
        try:
            r = subprocess.run(["conda", "env", "list", "--json"],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                envs = json.loads(r.stdout).get("envs", [])
                found = any(conda_env in e for e in envs)
                if not found:
                    print(f"[ENV CHECK] Conda environment '{conda_env}' NOT FOUND", file=sys.stderr)
                    return False, info
                print(f"[ENV CHECK] Conda environment '{conda_env}' found")
        except Exception as e:
            print(f"[ENV CHECK] Could not list conda envs: {e}", file=sys.stderr)

        # Check python3 is invocable
        try:
            r = subprocess.run(["conda", "run", "-n", conda_env, "python3", "-c",
                                "print('ok')"],
                               capture_output=True, text=True, timeout=15)
            info["python_ok"] = r.returncode == 0 and r.stdout.strip() == "ok"
            if not info["python_ok"]:
                print(f"[ENV CHECK] python3 in '{conda_env}' not reachable (rc={r.returncode}, stderr={r.stderr.strip()})", file=sys.stderr)
                return False, info
            print(f"[ENV CHECK] python3 in '{conda_env}' is reachable")
        except Exception as e:
            print(f"[ENV CHECK] Failed to exec python3 in '{conda_env}': {e}", file=sys.stderr)
            return False, info
    else:
        info["python_ok"] = True

    # Check torch
    snippet = (
        "import torch; "
        "print(torch.__version__); "
        "print(torch.cuda.is_available()); "
        "print(getattr(torch, 'xpu', None) and torch.xpu.is_available())"
    )
    base = [sys.executable]
    if conda_env:
        base = ["conda", "run", "-n", conda_env, "python3"]
    cmd = _isolated_cmd(base, snippet)

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            lines = r.stdout.strip().split("\n")
            info["torch_version"] = lines[0] if len(lines) > 0 else "unknown"
            info["cuda_available"] = lines[1].strip() == "True" if len(lines) > 1 else False
            info["xpu_available"] = lines[2].strip() == "True" if len(lines) > 2 else False
            info["torch_available"] = True
        else:
            print(f"[ENV CHECK] torch not importable (rc={r.returncode}): {r.stderr.strip()}", file=sys.stderr)
            return False, info
    except Exception as e:
        print(f"[ENV CHECK] torch probe failed: {e}", file=sys.stderr)
        return False, info

    print(f"[ENV CHECK] torch {info['torch_version']} available, CUDA={info['cuda_available']}, XPU={info['xpu_available']}")

    # Check pytest-timeout plugin (required for the per-case timeout below).
    # Fail fast rather than silently installing it, per this project's
    # fail-fast dependency policy (setup_env.sh owns environment bootstrap).
    base_pt = [sys.executable]
    if conda_env:
        base_pt = ["conda", "run", "-n", conda_env, "python3"]
    pt_cmd = _isolated_cmd(base_pt, "import pytest_timeout")
    try:
        r = subprocess.run(pt_cmd, capture_output=True, text=True, timeout=15)
        info["pytest_timeout_available"] = r.returncode == 0
    except Exception:
        info["pytest_timeout_available"] = False
    if not info["pytest_timeout_available"]:
        print(
            "[ENV CHECK] pytest-timeout plugin NOT installed in this environment. "
            "Per-case timeout enforcement requires it. Install with: "
            f"{'conda run -n ' + conda_env + ' ' if conda_env else ''}"
            "pip install pytest-timeout",
            file=sys.stderr,
        )
        return False, info
    print("[ENV CHECK] pytest-timeout plugin available")

    return True, info


def main():
    if sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    tasks_path = sys.argv[1]
    output_path = "results.json"
    log_dir = "test_logs"
    test_timeout = 600
    conda_env = None
    pytorch_root = os.environ.get("PYTORCH_FOLDER") or os.getcwd()

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--log-dir" and i + 1 < len(sys.argv):
            log_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--timeout" and i + 1 < len(sys.argv):
            test_timeout = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--env" and i + 1 < len(sys.argv):
            conda_env = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--pytorch-root" and i + 1 < len(sys.argv):
            pytorch_root = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown flag: {sys.argv[i]}", file=sys.stderr)
            sys.exit(1)

    pytorch_root = os.path.abspath(pytorch_root)
    if not os.path.isdir(pytorch_root):
        print(f"[error] --pytorch-root not a directory: {pytorch_root}", file=sys.stderr)
        sys.exit(1)
    print(f"PyTorch root: {pytorch_root}")

    with open(tasks_path) as f:
        data = json.load(f)

    tasks = data.get("tasks", [])
    already_resolved = data.get("already_resolved", [])

    print(f"Tasks needing classification: {len(tasks)}")
    print(f"Already resolved: {len(already_resolved)}")

    blank_tasks = [t for t in tasks if not t.get("status_xpu", "")]
    other_tasks = [t for t in tasks if t.get("status_xpu", "")]
    print(f"Blank status_xpu tasks (will run): {len(blank_tasks)}")
    print(f"Non-blank status_xpu tasks (skip local run): {len(other_tasks)}")

    ok, env_info = check_environment(conda_env)
    torch_ver = env_info["torch_version"]
    detail = f"Local test PASSED (torch {torch_ver})"
    print(f"Local torch version: {torch_ver}")

    if not ok:
        print(f"\n[ENV CHECK FAILED] Cannot run tests — torch environment not available.", file=sys.stderr)
        results_output = {
            "results": tasks + already_resolved,
            "summary": {
                "total": len(tasks) + len(already_resolved),
                "local_passed": 0,
                "still_blank": len(blank_tasks),
                "non_blank_skip_local": len(other_tasks),
                "already_resolved": len(already_resolved),
                "env_error": torch_ver if torch_ver == "unknown" else f"CUDA={env_info['cuda_available']} XPU={env_info['xpu_available']}",
            },
        }
        with open(output_path, "w") as f:
            json.dump(results_output, f, indent=2)
        print(f"Fallback results written to: {output_path}")
        sys.exit(1)

    if not env_info["xpu_available"] and not env_info["cuda_available"]:
        print("[ENV CHECK] Neither XPU nor CUDA backend available — cannot run GPU tests. Exiting.", file=sys.stderr)
        results_output = {
            "results": tasks + already_resolved,
            "summary": {
                "total": len(tasks) + len(already_resolved),
                "local_passed": 0,
                "still_blank": len(blank_tasks),
                "non_blank_skip_local": len(other_tasks),
                "already_resolved": len(already_resolved),
                "env_error": "XPU unavailable, CUDA unavailable",
            },
        }
        with open(output_path, "w") as f:
            json.dump(results_output, f, indent=2)
        print(f"Fallback results written to: {output_path}")
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)

    local_passed = []
    still_blank = []

    tests_by_file = defaultdict(list)
    for t in blank_tasks:
        tf = t.get("testfile_cuda", "unknown")
        tests_by_file[tf].append(t)

    summary_path = os.path.join(log_dir, "run_summary.log")
    with open(summary_path, "w") as summary:
        summary.write(f"Torch version: {torch_ver}\n")
        summary.write(f"Test timeout: {test_timeout}s (pytest-timeout, --timeout_method=thread)\n\n")

    passed_count = 0
    failed_count = 0

    for tf, test_list in sorted(tests_by_file.items()):
        log_path = os.path.join(log_dir, tf.replace("/", "_") + ".log")
        test_names = list(set(t.get("name_cuda", "") for t in test_list))
        cls_names = list(set(t.get("classname_cuda", "") for t in test_list))

        with open(summary_path, "a") as summary:
            summary.write(f"\n{'='*60}\n")
            summary.write(f"File: {tf}\n")
            summary.write(f"Classes: {', '.join(cls_names)}\n")
            summary.write(f"Tests to run: {len(test_list)}\n")

        results_per_test = {}

        for t in test_list:
            name = t.get("name_cuda", "")
            cls_name = t.get("classname_cuda", "")
            # Use XPU names for the -k filter (tests are _xpu not _cuda)
            filter_name = t.get("name_xpu", name)
            filter_cls = t.get("classname_xpu", cls_name)
            test_path = os.path.abspath(tf) if os.path.isabs(tf) else os.path.join(pytorch_root, tf)

            if not os.path.exists(test_path):
                print(f"  SKIP {name} - file not found: {test_path}")
                results_per_test[name] = {"passed": False, "reason": "file_not_found"}
                continue

            run_env = os.environ.copy()
            existing_pp = run_env.get("PYTHONPATH", "")
            run_env["PYTHONPATH"] = (
                pytorch_root + os.pathsep + existing_pp if existing_pp else pytorch_root
            )

            base = [sys.executable]
            if conda_env:
                base = ["conda", "run", "-n", conda_env, "python3"]
            base_cmd = _isolated(base) + ["-m", "pytest", test_path,
                         "-v", "--no-header", "--tb=short",
                         "--timeout", str(test_timeout), "--timeout_method=thread"]

            k_exprs = [f"{filter_cls} and {filter_name}"]
            k_exprs.append(filter_name)
            base_clean = re.sub(r'_(cuda|xpu)$', '', filter_name)
            if base_clean != filter_name:
                k_exprs.append(base_clean)
            if name != filter_name:
                k_exprs.append(name)
                base_clean2 = re.sub(r'_(cuda|xpu)$', '', name)
                if base_clean2 != name:
                    k_exprs.append(base_clean2)
            if '_prepacked' in filter_name or '_aten_' in filter_name:
                for sep in ['_aten_', '_prepacked']:
                    idx = filter_name.find(sep)
                    if idx > 0:
                        base_part = filter_name[:idx]
                        param_part = filter_name[idx:]
                        k_exprs.append(f"{base_part} and {param_part}")
                        k_exprs.append(f"{base_part}[{param_part}]")
            seen = set()
            k_exprs_unique = [e for e in k_exprs if not (e in seen or seen.add(e))]

            start = time.time()
            p = None
            chosen_k = k_exprs_unique[0]
            for ke in k_exprs_unique:
                cmd_list = base_cmd + ["-k", ke]
                try:
                    r = subprocess.run(
                        cmd_list, capture_output=True, text=True,
                        timeout=test_timeout + WATCHDOG_BUFFER_SECONDS, cwd=pytorch_root, env=run_env
                    )
                except subprocess.TimeoutExpired:
                    elapsed = time.time() - start
                    msg = f"TIMEOUT after {elapsed:.0f}s"
                    print(f"  {msg}: {name}")
                    with open(log_path, "a") as lf:
                        lf.write(f"\n--- {name} --- {msg}\n")
                    results_per_test[name] = {"passed": False, "reason": "timeout"}
                    continue
                has_test_errors = "error" in r.stderr.lower() and "conda" not in r.stderr.lower()
                if "0 selected" not in r.stdout or has_test_errors:
                    p = r
                    chosen_k = ke
                    if ke != k_exprs_unique[0]:
                        elapsed_extra = time.time() - start
                        with open(log_path, "a") as lf:
                            lf.write(f"\n[FALLBACK] '{k_exprs_unique[0]}' matched 0 for '{name}'; retried with '{ke}' (took {elapsed_extra:.1f}s).\n")
                            lf.write(r.stdout)
                            if r.stderr:
                                lf.write("\nSTDERR:\n")
                                lf.write(r.stderr)
                    break

            if p is None:
                elapsed = time.time() - start
                with open(log_path, "a") as lf:
                    lf.write(f"\n--- {name} ({elapsed:.1f}s) --- FAILED/SKIP ---\n")
                    lf.write("No matching tests found with any -k expression.\n")
                print(f"  FAIL ({elapsed:.1f}s): {name} (no matching test)")
                results_per_test[name] = {"passed": False, "reason": "no_match"}
                continue

            elapsed = time.time() - start

            our_test_passed = False
            test_refs = [name, filter_name, base_clean]
            for line in p.stdout.split("\n"):
                if "PASSED" in line:
                    for ref in test_refs:
                        if ref and ref in line:
                            our_test_passed = True
                            break
                if our_test_passed:
                    break
            all_passed = "FAILED" not in p.stdout and "ERROR" not in p.stderr if p.stderr else "FAILED" not in p.stdout
            matched_ours = our_test_passed or (chosen_k != k_exprs_unique[0])

            passed = (our_test_passed and all_passed)

            status = "PASSED" if passed else "FAILED/SKIP"

            with open(log_path, "a") as lf:
                lf.write(f"\n--- {name} ({elapsed:.1f}s) --- {status} ---\n")
                lf.write(p.stdout)
                if p.stderr:
                    lf.write("\nSTDERR:\n")
                    lf.write(p.stderr)

            if passed:
                passed_count += 1
                results_per_test[name] = {"passed": True, "reason": "local_pass"}
                print(f"  PASS ({elapsed:.1f}s): {name}")
            else:
                failed_count += 1
                err_line = "FAIL"
                for line in p.stdout.split("\n"):
                    if "FAILED" in line:
                        err_line = line.strip()[:150]
                    elif "ERROR" in line and err_line == "FAIL":
                        err_line = line.strip()[:150]
                results_per_test[name] = {"passed": False, "reason": err_line}
                print(f"  FAIL ({elapsed:.1f}s): {name}")

        with open(summary_path, "a") as summary:
            file_passed = sum(1 for v in results_per_test.values() if v["passed"])
            file_total = len(results_per_test)
            summary.write(f"Results: {file_passed}/{file_total} passed\n")

        for t in test_list:
            name = t.get("name_cuda", "")
            r = results_per_test.get(name, {"passed": False, "reason": "not_run"})
            if r["passed"]:
                entry = {
                    "testfile_cuda": t.get("testfile_cuda", ""),
                    "classname_cuda": t.get("classname_cuda", ""),
                    "name_cuda": name,
                    "Reason": "Local Passed",
                    "DetailReason": detail,
                    "ReuseSource": "",
                }
                local_passed.append(entry)
            else:
                still_blank.append(t)

    with open(summary_path, "a") as summary:
        summary.write(f"\n{'='*60}\n")
        summary.write(f"Total local passed: {passed_count}\n")
        summary.write(f"Total not passed (still blank): {len(still_blank)}\n")
        summary.write(f"Total non-blank tasks (not run): {len(other_tasks)}\n")

    print(f"\n{'='*60}")
    print(f"Local passed: {passed_count}")
    print(f"Still blank (for cascade): {len(still_blank)}")
    print(f"Summary log: {summary_path}")

    results_output = {
        "results": local_passed + still_blank + other_tasks + already_resolved,
        "summary": {
            "total": len(tasks) + len(already_resolved),
            "local_passed": passed_count,
            "still_blank": len(still_blank),
            "non_blank_skip_local": len(other_tasks),
            "already_resolved": len(already_resolved),
        },
    }

    with open(output_path, "w") as f:
        json.dump(results_output, f, indent=2)

    print(f"Results written to: {output_path}")
    print(f"Test logs: {log_dir}/")


if __name__ == "__main__":
    main()
