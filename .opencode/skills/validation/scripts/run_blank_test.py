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

Results written to results.json can be fed into write_results.py for Excel output.

Usage:
    python3 run_blank_test.py <tasks.json> [--output results.json] [--log-dir test_logs]
                             [--timeout SECONDS] [--env CONDA_ENV_NAME]

Examples:
    python3 run_blank_test.py tasks.json --output results.json
    python3 run_blank_test.py tasks.json --output results.json --timeout 300
"""

import json
import os
import subprocess
import sys
import time
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    test_timeout = 300
    conda_env = None

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
        else:
            print(f"Unknown flag: {sys.argv[i]}", file=sys.stderr)
            sys.exit(1)

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
        summary.write(f"Test timeout: {test_timeout}s\n\n")

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
            test_path = os.path.abspath(tf) if os.path.isabs(tf) else os.path.join(os.getcwd(), tf)

            if not os.path.exists(test_path):
                print(f"  SKIP {name} - file not found: {test_path}")
                results_per_test[name] = {"passed": False, "reason": "file_not_found"}
                continue

            pypath = f"PYTHONPATH={os.getcwd()}" if os.getenv("PYTHONPATH") else ""

            k_expr = f"{filter_cls} and {filter_name}"
            base = [sys.executable]
            if conda_env:
                base = ["conda", "run", "-n", conda_env, "python3"]
            cmd_list = _isolated(base) + ["-m", "pytest", test_path,
                         "-k", k_expr, "-v", "--no-header", "--tb=short"]

            start = time.time()
            try:
                p = subprocess.run(
                    cmd_list, capture_output=True, text=True,
                    timeout=test_timeout, cwd=os.getcwd()
                )
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                msg = f"TIMEOUT after {elapsed:.0f}s"
                print(f"  {msg}: {name}")
                with open(log_path, "a") as lf:
                    lf.write(f"\n--- {name} --- {msg}\n")
                results_per_test[name] = {"passed": False, "reason": "timeout"}
                continue

            elapsed = time.time() - start
            passed = "1 passed" in p.stdout and "FAILED" not in p.stdout
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
