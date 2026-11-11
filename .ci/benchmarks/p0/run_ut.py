# Copyright 2020-2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Run P0 UT groups from ut.yml in an already prepared environment.

Example: python run_ut.py --pytorch-root /work/pytorch inductor non-inductor
Use --dry-run to inspect commands. Child output is written directly to per-file
logs; follow a log with tail -f. PYTEST_ADDOPTS is inherited without modification.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import yaml


class ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass
class Task:
    group: str
    repo: str
    path: str
    cwd: Path
    command: list[str]
    log: Path
    junit: Path


def require_mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def build_plan(config, selected, roots, output, prefix, run_id):
    require_mapping(config, "Config")
    if set(config) != {"groups"}:
        raise ValueError("Config must contain only groups")
    groups = require_mapping(config["groups"], "groups")
    selected = selected or list(groups)
    if len(set(selected)) != len(selected) or any(group not in groups for group in selected):
        raise ValueError(f"Select distinct groups from: {', '.join(map(str, groups))}")
    tasks = []
    for group, repos in groups.items():
        if not isinstance(group, str) or not group or not all(c.isalnum() or c in "-_" for c in group):
            raise ValueError(f"Invalid group name: {group!r}")
        for repo, files in require_mapping(repos, group).items():
            if repo not in roots:
                raise ValueError(f"Unknown repository: {repo!r}")
            root = roots[repo]
            for path, options in require_mapping(files, repo).items():
                if (
                    not isinstance(path, str)
                    or not path.endswith(".py")
                    or Path(path).is_absolute()
                    or ".." in Path(path).parts
                    or any(c in path for c in "\r\n\0:")
                ):
                    raise ValueError(f"Invalid test path: {path!r}")
                require_mapping(options, path)
                if set(options) - {"skip"}:
                    raise ValueError(f"Unknown options for {path}: {set(options)}")
                cases = options.get("skip", [])
                if not isinstance(cases, list) or any(
                    not isinstance(case, str) or not case.strip() or any(c in case for c in "\r\n\0") for case in cases
                ):
                    raise ValueError(f"Invalid skip cases for {path}")
                if group not in selected:
                    continue
                if not (root / path).is_file():
                    raise ValueError(f"Test file does not exist: {root / path}")
                name = f"{prefix}{group}__{repo}__{path.replace('/', '__')}_{run_id}"
                junit = output / f"{name}.xml"
                command = ["pytest", path, f"--rootdir={root}", f"--junit-xml={junit}"] + [
                    f"--deselect={path}::{case}" for case in cases
                ]
                tasks.append(
                    Task(
                        group,
                        repo,
                        path,
                        root,
                        command,
                        output / f"{name}.log",
                        junit,
                    )
                )
    # Respect CLI group order, and YAML declaration order within each group.
    tasks.sort(key=lambda task: selected.index(task.group))
    if not tasks:
        raise ValueError("Selected groups contain no tests")
    return tasks


def run_task(task, timeout):
    started = time.monotonic()
    result = {**asdict(task), "status": "failed", "returncode": None}
    print(
        f"\n[{task.group}/{task.repo}] {task.path}\ncwd: {task.cwd}\n"
        f"command: {shlex.join(task.command)}\nlog: {task.log}",
        flush=True,
    )
    with task.log.open("w", encoding="utf-8") as log:
        try:
            with subprocess.Popen(
                task.command,
                cwd=task.cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            ) as process:
                try:
                    result["returncode"] = process.wait(timeout=timeout or None)
                    result["status"] = "passed" if process.returncode == 0 else "failed"
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    result["returncode"] = process.wait()
                    result["status"] = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
        except OSError as error:
            result["error"] = str(error)
            log.write(f"ERROR: {error}\n")
    try:
        tree = ET.parse(task.junit).getroot()
        suites = [tree] if tree.tag == "testsuite" else tree.findall("testsuite")
        result["counts"] = {
            key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")
        }
        if result["status"] == "passed" and (
            not result["counts"]["tests"] or result["counts"]["failures"] or result["counts"]["errors"]
        ):
            result["status"] = "failed"
    except (OSError, ET.ParseError, ValueError) as error:
        result["error"] = f"JUnit report unavailable or invalid: {error}"
        if result["status"] == "passed":
            result["status"] = "failed"
    result["seconds"] = round(time.monotonic() - started, 3)
    print(
        f"{result['status'].upper()} ({result['seconds']}s, exit={result['returncode']})",
        flush=True,
    )
    return result


def write_summary(path, results, total):
    path.with_suffix(".json").write_text(
        json.dumps(
            {"planned": total, "completed": len(results), "results": results},
            indent=2,
            default=str,
        )
        + "\n"
    )
    rows = [
        "| Group | Repo | File | Status | Exit | Seconds |",
        "|---|---|---|---|---:|---:|",
    ]
    for result in results:
        rows.append(
            "| "
            + " | ".join(
                str(result[key])
                for key in (
                    "group",
                    "repo",
                    "path",
                    "status",
                    "returncode",
                    "seconds",
                )
            )
            + " |"
        )
    path.with_suffix(".md").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="*", help="Groups to run; default: all groups in YAML order")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("ut.yml"))
    parser.add_argument("--pytorch-root", type=Path, default=os.environ.get("PYTORCH_ROOT_DIR"))
    parser.add_argument(
        "--xpu-ops-root",
        type=Path,
        help="Default: PYTORCH_ROOT_DIR/third_party/torch-xpu-ops",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.environ.get("XML_OUTPUT_DIR", "test-reports"),
    )
    parser.add_argument("--xml-prefix", default=os.environ.get("XML_PREFIX", "op_p0_with_"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=0,
        help="Per-file timeout in seconds; 0 disables it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without executing",
    )
    args = parser.parse_args()
    if args.pytorch_root is None:
        parser.error("--pytorch-root or PYTORCH_ROOT_DIR is required")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and nonnegative")
    if any(c in args.xml_prefix for c in "/\\\r\n\0"):
        parser.error("--xml-prefix must be a filename prefix")
    roots = {
        "pytorch": args.pytorch_root.resolve(),
        "xpu_ops": (args.xpu_ops_root or args.pytorch_root / "third_party/torch-xpu-ops").resolve(),
    }
    output = args.output_dir.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        config_text = args.config.read_text(encoding="utf-8")
        config = yaml.load(config_text, Loader=ConfigLoader)
        tasks = build_plan(config, args.groups, roots, output, args.xml_prefix, run_id)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps([asdict(task) for task in tasks], indent=2, default=str))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = None
    manifest = {
        "python": sys.version,
        "torch": torch_version,
        "roots": roots,
        "config": config_text,
        "tasks": [asdict(task) for task in tasks],
    }
    (output / f"manifest_{run_id}.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    summary = output / f"summary_{run_id}"
    results = []
    write_summary(summary, results, len(tasks))
    for task in tasks:
        result = run_task(task, args.timeout)
        results.append(result)
        write_summary(summary, results, len(tasks))
        if result["status"] == "interrupted":
            return 130
    print(
        f"\nCompleted {len(results)} files. Summary: {summary.with_suffix('.md')}",
        flush=True,
    )
    return int(any(result["status"] != "passed" for result in results))


if __name__ == "__main__":
    sys.exit(main())
