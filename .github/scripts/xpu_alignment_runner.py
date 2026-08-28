#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Execute immutable XPU alignment reproducers outside an agent step."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pwd
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from xpu_alignment_collect import CollectionError, sha256, validate_collection


UNIT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_TIMEOUT_SECONDS = 120
PR_SET_CHILD_SUBREAPER = 36
ENVIRONMENT_FIELDS = {
    "python_executable",
    "python_version",
    "torch_version",
    "torch_path",
    "xpu_available",
    "xpu_device_name",
    "environment_warnings",
}


class PlanError(ValueError):
    pass


def _inside(root: Path, value: object, *, existing: bool) -> Path:
    relative = Path(str(value or ""))
    if not str(value or "") or relative.is_absolute() or ".." in relative.parts:
        raise PlanError(f"unsafe relative path: {value!r}")
    candidate = root / relative
    if candidate.is_symlink():
        raise PlanError(f"symlinks are not allowed: {relative}")
    resolved = candidate.resolve(strict=existing)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PlanError(f"path escapes run root: {relative}") from error
    return resolved


def load_prepare(root: Path, prepare_path: Path) -> list[dict[str, object]]:
    try:
        prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot read prepare artifact: {error}") from error
    if not isinstance(prepare, dict) or prepare.get("schema_version") != 1:
        raise PlanError("prepare artifact uses an unsupported schema")
    if prepare.get("status") != "complete":
        raise PlanError("prepare artifact must be complete before execution")
    blockers = prepare.get("blockers")
    if not isinstance(blockers, list) or blockers:
        raise PlanError("prepare artifact must have an empty blocker list")
    try:
        collection_path, collection, inventory = validate_collection(
            root,
            {
                "start": str(prepare["scan_window"]["start"]),
                "end": str(prepare["scan_window"]["end"]),
            },
        )
    except (CollectionError, KeyError, TypeError, ValueError) as error:
        raise PlanError(f"collection artifact rejected: {error}") from error
    if prepare.get("collection_sha256") != sha256(collection_path):
        raise PlanError("prepare collection digest does not match")
    if prepare.get("collection_status") != collection.get("status"):
        raise PlanError("prepare collection status does not match")
    executions = prepare.get("executions")
    if not isinstance(executions, list):
        raise PlanError("prepare executions must be a list")

    validated: set[str] = set()
    decisions = prepare.get("decisions")
    if not isinstance(decisions, list):
        raise PlanError("prepare decisions must be a list")
    decision_ids: set[str] = set()
    for index, item in enumerate(decisions):
        if not isinstance(item, dict):
            raise PlanError(f"decision {index} is not an object")
        unit_id = item.get("id")
        if (
            not isinstance(unit_id, str)
            or not UNIT_ID_RE.fullmatch(unit_id)
            or unit_id in decision_ids
            or unit_id not in inventory
        ):
            raise PlanError(f"invalid or duplicate decision id: {unit_id!r}")
        triage = item.get("triage")
        if triage not in {"reject", "validate"}:
            raise PlanError(f"{unit_id}: invalid triage")
        if not str(item.get("reason", "")).strip():
            raise PlanError(f"{unit_id}: missing triage reason")
        decision_ids.add(unit_id)
        if triage == "validate":
            validated.add(unit_id)
    if decision_ids != set(inventory):
        raise PlanError("decision coverage does not match collected inventory")

    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, entry in enumerate(executions):
        if not isinstance(entry, dict):
            raise PlanError(f"execution {index} is not an object")
        unit_id = entry.get("id")
        if not isinstance(unit_id, str) or not UNIT_ID_RE.fullmatch(unit_id) or unit_id in seen:
            raise PlanError(f"invalid or duplicate unit id: {unit_id!r}")
        script = _inside(root, entry.get("script"), existing=True)
        expected_digest = entry.get("script_sha256")
        if not isinstance(expected_digest, str) or not SHA256_RE.fullmatch(expected_digest):
            raise PlanError(f"{unit_id}: invalid script digest")
        if sha256(script) != expected_digest:
            raise PlanError(f"{unit_id}: script digest does not match")
        timeout = entry.get("timeout_seconds")
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or not 1 <= timeout <= MAX_TIMEOUT_SECONDS
        ):
            raise PlanError(f"{unit_id}: invalid timeout")
        for field in ("oracle", "target_path"):
            if not str(entry.get(field, "")).strip():
                raise PlanError(f"{unit_id}: missing {field}")
        seen.add(unit_id)
        normalized.append(
            {
                "id": unit_id,
                "script": script,
                "script_path": str(script.relative_to(root)),
                "script_sha256": expected_digest,
                "timeout_seconds": timeout,
            }
        )
    if seen != validated:
        raise PlanError("execution coverage does not match validated inventory")
    return normalized


def _safe_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CMAKE_PREFIX_PATH",
        "ONEAPI_ROOT",
        "ZE_AFFINITY_MASK",
        "SYCL_DEVICE_FILTER",
        "LANG",
        "LC_ALL",
        "TZ",
    }
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _execution_context(
    scratch: str, identity: tuple[int, int, list[int]] | None
) -> tuple[dict[str, str], dict[str, object]]:
    scratch_path = Path(scratch)
    debug = scratch_path / "torch_compile_debug"
    debug.mkdir()
    environment = _safe_environment()
    environment.update(
        {
            "HOME": scratch,
            "TMPDIR": scratch,
            "TORCH_COMPILE_DEBUG_DIR": str(debug),
        }
    )
    process: dict[str, object] = {}
    if identity is not None:
        uid, gid, groups = identity
        os.chown(scratch_path, uid, gid)
        os.chown(debug, uid, gid)
        environment.update({"USER": str(uid), "LOGNAME": str(uid)})
        process.update(
            {
                "user": uid,
                "group": gid,
                "extra_groups": groups,
                "umask": 0o077,
            }
        )
    return environment, process


def _validated_environment(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != ENVIRONMENT_FIELDS:
        raise PlanError("environment probe returned an invalid field set")
    for field in ("python_executable", "python_version", "torch_version"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise PlanError(f"environment probe returned invalid {field}")
    for field in ("torch_path", "xpu_device_name"):
        if value.get(field) is not None and not isinstance(value[field], str):
            raise PlanError(f"environment probe returned invalid {field}")
    warnings = value.get("environment_warnings")
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) and item for item in warnings
    ):
        raise PlanError("environment probe returned invalid warnings")
    if value.get("xpu_available") is not True:
        raise PlanError("XPU is unavailable in the reproducer environment")
    return value


def probe_environment(
    python: Path, identity: tuple[int, int, list[int]] | None = None
) -> dict[str, object]:
    """Record the actual credential-free environment used by reproducers."""
    source = """
import json
import platform
import sys

payload = {
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": "unknown",
    "torch_path": None,
    "xpu_available": False,
    "xpu_device_name": None,
    "environment_warnings": [],
}
try:
    import torch
except Exception as error:
    payload["core_error"] = f"cannot import torch: {type(error).__name__}: {error}"
else:
    payload["torch_version"] = str(getattr(torch, "__version__", "unknown"))
    try:
        path = torch.__file__
        payload["torch_path"] = str(path) if path else None
        if not path:
            payload["environment_warnings"].append("torch path is unavailable")
    except Exception as error:
        payload["environment_warnings"].append(
            f"torch path unavailable: {type(error).__name__}: {error}"
        )
    try:
        payload["xpu_available"] = bool(torch.xpu.is_available())
    except Exception as error:
        payload["core_error"] = f"cannot query XPU availability: {type(error).__name__}: {error}"
    if payload["xpu_available"]:
        try:
            payload["xpu_device_name"] = str(torch.xpu.get_device_name(0))
        except Exception as error:
            payload["environment_warnings"].append(
                f"XPU device name unavailable: {type(error).__name__}: {error}"
            )
print(json.dumps(payload, sort_keys=True))
"""
    with tempfile.TemporaryDirectory(prefix="xpu-alignment-environment-") as scratch:
        environment, process = _execution_context(scratch, identity)
        try:
            completed = subprocess.run(
                [str(python), "-I", "-c", source],
                cwd=scratch,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                **process,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise PlanError(f"environment probe failed: {error}") from error
    if completed.returncode != 0:
        raise PlanError(
            f"environment probe exited {completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise PlanError("environment probe returned malformed JSON") from error
    if isinstance(payload, dict) and payload.get("core_error"):
        raise PlanError(str(payload["core_error"]))
    return _validated_environment(payload)


def _identity(user: str) -> tuple[int, int, list[int]]:
    try:
        account = pwd.getpwnam(user)
    except KeyError as error:
        raise PlanError(f"execution user does not exist: {user}") from error
    if account.pw_uid == 0:
        raise PlanError("reproducers must not run as root")
    if os.geteuid() != 0 and account.pw_uid != os.geteuid():
        raise PlanError("changing execution identity requires root")
    return account.pw_uid, account.pw_gid, os.getgrouplist(user, account.pw_gid)


def _become_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise PlanError(f"cannot establish reproducer process boundary: {os.strerror(error)}")


def _direct_children() -> list[int]:
    children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    try:
        return [int(value) for value in children.read_text(encoding="ascii").split()]
    except OSError as error:
        raise PlanError(f"cannot inspect reproducer descendants: {error}") from error


def _reap_children() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _terminate_descendants() -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        children = _direct_children()
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _reap_children()
        if not _direct_children():
            return
        time.sleep(0.01)
    raise PlanError("reproducer descendants survived cleanup")


def run_plan(
    root: Path,
    python: Path,
    prepare_path: Path,
    entries: list[dict[str, object]],
    identity: tuple[int, int, list[int]] | None = None,
    *,
    environment: dict[str, object] | None = None,
) -> dict[str, object]:
    if not python.is_file():
        raise PlanError(f"python executable does not exist: {python}")
    _become_child_subreaper()
    recorded_environment = _validated_environment(
        environment if environment is not None else probe_environment(python, identity)
    )
    logs = root / "runner/logs"
    logs.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for entry in entries:
        unit_id = str(entry["id"])
        script = Path(entry["script"])
        log = logs / f"{unit_id}.log"
        started = time.monotonic()
        returncode: int | None = None
        timed_out = False
        error: str | None = None
        with log.open("wb") as output, tempfile.TemporaryDirectory(
            prefix=f"xpu-alignment-{unit_id}-"
        ) as scratch:
            child_environment, process_options = _execution_context(scratch, identity)
            if sha256(script) != entry["script_sha256"]:
                error = "script bytes changed after prepare validation"
            else:
                process: subprocess.Popen[bytes] | None = None
                try:
                    process = subprocess.Popen(
                        [str(python), "-I", str(script)],
                        cwd=scratch,
                        env=child_environment,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        **process_options,
                    )
                    returncode = process.wait(timeout=int(entry["timeout_seconds"]))
                except subprocess.TimeoutExpired:
                    timed_out = True
                except (OSError, subprocess.SubprocessError) as exc:
                    error = str(exc)
                finally:
                    if process is not None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    _terminate_descendants()
        results.append(
            {
                "id": unit_id,
                "script_sha256": entry["script_sha256"],
                "command": [str(python), "-I", entry["script_path"]],
                "log": str(log.relative_to(root)),
                "log_sha256": sha256(log),
                "returncode": returncode,
                "timed_out": timed_out,
                "duration_seconds": round(time.monotonic() - started, 3),
                "error": error,
            }
        )
    return {
        "schema_version": 1,
        "collection_sha256": json.loads(prepare_path.read_text(encoding="utf-8"))[
            "collection_sha256"
        ],
        "prepare_sha256": sha256(prepare_path),
        "status": "complete",
        "environment": recorded_environment,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--user", required=True)
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        prepare = args.prepare if args.prepare.is_absolute() else root / args.prepare
        prepare = prepare.resolve(strict=True)
        prepare.relative_to(root)
        results = args.results if args.results.is_absolute() else root / args.results
        results = results.resolve()
        results.relative_to(root)
        entries = load_prepare(root, prepare)
        payload = run_plan(root, args.python.resolve(), prepare, entries, _identity(args.user))
    except (OSError, PlanError, ValueError) as error:
        print(f"prepare artifact rejected: {error}")
        return 2
    results.parent.mkdir(parents=True, exist_ok=True)
    temporary = results.with_suffix(results.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(results)
    print(json.dumps({"executed": len(entries), "results": str(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
