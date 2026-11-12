# Copyright 2020-2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Run P0 UT groups from ut.yml in an already prepared environment.

Example: python run_ut.py --pytorch-root /work/pytorch inductor non-inductor
Use --dry-run to inspect commands. Child output is written directly to per-file
logs; follow a log with tail -f. PYTEST_ADDOPTS is inherited without modification.
There is no per-file timeout; bound the whole run from the caller (CI step).
Exit codes: 0 all passed, 1 test failures, 2 usage/config error, 3 runner
error, 130 interrupted. A file that fails without a failing case in its JUnit
(crash, interrupt, no tests) gets a <name>_runner.xml error record.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
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
                command = [sys.executable, "-m", "pytest", path, f"--rootdir={root}", f"--junit-xml={junit}"] + [
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


def run_task(task):
    started = time.monotonic()
    result = {**asdict(task), "status": "failed", "returncode": None}
    print(
        f"\n[{task.group}/{task.repo}] {task.path}\ncwd: {task.cwd}\n"
        f"command: {shlex.join(task.command)}\nlog: {task.log}",
        flush=True,
    )
    with task.log.open("w", encoding="utf-8") as log:
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if task.repo == "pytorch" and Path(task.path).parent == Path("test/distributed/pipelining"):
                sibling_dir = task.cwd / Path(task.path).parent
                env["PYTHONPATH"] = os.pathsep.join(
                    filter(None, (str(sibling_dir), env.get("PYTHONPATH", "")))
                )
            with subprocess.Popen(
                task.command,
                cwd=task.cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            ) as process:
                try:
                    result["returncode"] = process.wait()
                    result["status"] = "passed" if process.returncode == 0 else "failed"
                except KeyboardInterrupt:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    result["returncode"] = process.wait()
                    result["status"] = "interrupted"
        except OSError as error:
            result["error"] = str(error)
            log.write(f"ERROR: {error}\n")
    try:
        tree = ET.parse(task.junit).getroot()
        suites = [tree] if tree.tag == "testsuite" else tree.findall("testsuite")
        result["counts"] = {
            key: sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")
        }
        result["failed_cases"] = failed_cases(tree, task.path)
        if result["status"] == "passed" and (
            not result["counts"]["tests"] or result["counts"]["failures"] or result["counts"]["errors"]
        ):
            result["status"] = "failed"
    except (OSError, ET.ParseError, ValueError) as error:
        result["error"] = f"JUnit report unavailable or invalid: {error}"
        if result["status"] == "passed":
            result["status"] = "failed"
    result["seconds"] = round(time.monotonic() - started, 3)
    if result["status"] != "passed" and not result.get("failed_cases"):
        write_file_error(task, result)
    print(
        f"{result['status'].upper()} ({result['seconds']}s, exit={result['returncode']})",
        flush=True,
    )
    return result


def show_tail(log, lines=50):
    """Echo the end of a failing file's log so CI consoles show the cause."""
    try:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return
    print(f"----- last {len(tail)} lines of {log} -----", *tail, "-----", sep="\n", flush=True)


def failed_cases(tree, path):
    """Pytest node IDs (relative to the task root) of failed/errored cases."""
    nodes = []
    for case in tree.iter("testcase"):
        if case.find("failure") is None and case.find("error") is None:
            continue
        name = case.get("name", "")
        cls = case.get("classname", "").rsplit(".", 1)[-1]
        if not cls or not name:
            # Collection error (empty classname): rerun the whole file.
            nodes.append(path)
        elif cls == Path(path).stem:
            # Module-level test function: classname is the module path.
            nodes.append(f"{path}::{name}")
        else:
            nodes.append(f"{path}::{cls}::{name}")
    return list(dict.fromkeys(nodes))


def write_file_error(task, result):
    """Record a file that failed without any failed case in its JUnit.

    A crash, an interrupt, an internal pytest error or an empty run leaves no
    failing <testcase>, so check-ut.py and ut_result_check.sh, which only read
    JUnit, would not see it. Emit the shape pytest uses for a collection error
    (empty classname, dotted module name) so it is reported against the file.
    """
    module = task.path[:-3].replace("/", ".")
    reason = f"pytest exited {result['returncode']} ({result['status']})"
    if not task.junit.is_file():
        reason += ", no JUnit report (crashed or killed)"
    suite = ET.Element("testsuite", name="p0-runner", tests="1", errors="1", failures="0", skipped="0")
    case = ET.SubElement(suite, "testcase", classname="", name=module, time=str(result["seconds"]))
    ET.SubElement(case, "error", message=f"{reason}; see {task.log.name}").text = reason
    ET.ElementTree(suite).write(
        task.junit.with_name(task.junit.stem + "_runner.xml"), encoding="utf-8", xml_declaration=True
    )


def display_root(root):
    """Root relative to the CI workspace, like other categories' hints."""
    base = os.environ.get("GITHUB_WORKSPACE")
    if base:
        try:
            return str(Path(root).relative_to(Path(base).resolve()))
        except ValueError:
            pass
    return str(root)


def write_reproduce(path, tasks, results):
    """Reproduce hints, rewritten after every file.

    The File Path/Reproduce Command header is what ut_collect_evidence.py
    parses. Node IDs are relative to each repository root, so every root is
    listed and each failed case gets a ready-to-run command in its own root.
    """
    roots = {t.repo: t.cwd for t in tasks}
    env = " ".join(f"{k}={shlex.quote(os.environ[k])}" for k in ("PYTORCH_TEST_WITH_SLOW",) if k in os.environ)
    prefix = f"{env} " if env else ""
    lines = [
        f"File Path: cd {display_root(next(iter(roots.values())))}",
        "Reproduce Command: pytest -sv failed_case",
    ]
    lines += [f"Root ({repo}): cd {display_root(root)}" for repo, root in roots.items()]
    failed = [(r, node) for r in results for node in r.get("failed_cases", [])]
    failed += [(r, r["path"]) for r in results if r["status"] != "passed" and not r.get("failed_cases")]
    if failed:
        lines.append("Failed cases:")
        lines += [
            f"cd {display_root(r['cwd'])} && {prefix}python -m pytest -sv {shlex.quote(node)}"
            for r, node in failed
        ]
    path.write_text("\n".join(lines) + "\n")


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
        "--reproduce-log",
        type=Path,
        help="Write per-repository reproduce commands to this file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without executing",
    )
    args = parser.parse_args()
    if args.pytorch_root is None:
        parser.error("--pytorch-root or PYTORCH_ROOT_DIR is required")
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
    if args.reproduce_log:
        write_reproduce(args.reproduce_log, tasks, [])
    results = []
    write_summary(summary, results, len(tasks))
    try:
        return run_all(tasks, summary, results, args.reproduce_log)
    except KeyboardInterrupt:
        # A signal outside a running test (e.g. while writing reports).
        print("Interrupted", flush=True)
        return 130
    except Exception:
        # Exit 1 means test failures; a runner bug must not look like one.
        traceback.print_exc()
        return 3


def run_all(tasks, summary, results, reproduce_log):
    for task in tasks:
        result = run_task(task)
        if reproduce_log:
            write_reproduce(reproduce_log, tasks, results + [result])
        results.append(result)
        write_summary(summary, results, len(tasks))
        if result["status"] != "passed":
            show_tail(task.log)
        if result["status"] == "interrupted":
            return 130
    print(
        f"\nCompleted {len(results)} files. Summary: {summary.with_suffix('.md')}",
        flush=True,
    )
    return int(any(result["status"] != "passed" for result in results))


if __name__ == "__main__":
    sys.exit(main())
