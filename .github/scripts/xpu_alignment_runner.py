#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Execute an XPU alignment repro plan outside the credential-bearing agent step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path


UNIT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_TIMEOUT_SECONDS = 600


class PlanError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, value: object, *, existing: bool) -> Path:
    text = str(value or "")
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise PlanError(f"unsafe relative path: {text!r}")
    candidate = root / relative
    if candidate.is_symlink():
        raise PlanError(f"symlinks are not allowed: {text}")
    resolved = candidate.resolve(strict=existing)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PlanError(f"path escapes run root: {text}") from exc
    return resolved


def _safe_environment() -> dict[str, str]:
    """Strip credentials defensively even though the runner step should receive none."""
    sensitive_fragments = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "PRIVATE_KEY")
    blocked_prefixes = ("AWS_", "GITHUB_", "ACTIONS_")
    blocked_names = {"GH_TOKEN", "SSH_AUTH_SOCK"}
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in blocked_names
        and not name.startswith(blocked_prefixes)
        and not any(fragment in name.upper() for fragment in sensitive_fragments)
    }
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _identity(user: str) -> tuple[int, int, list[int]]:
    try:
        account = pwd.getpwnam(user)
    except KeyError as exc:
        raise PlanError(f"execution user does not exist: {user}") from exc
    if account.pw_uid == 0:
        raise PlanError("reproducers must not run as root")
    if os.geteuid() != 0 and account.pw_uid != os.geteuid():
        raise PlanError("changing execution identity requires root")
    return account.pw_uid, account.pw_gid, os.getgrouplist(user, account.pw_gid)


def _drop_privileges(uid: int, gid: int, groups: list[int]) -> None:
    os.setgroups(groups)
    os.setgid(gid)
    os.setuid(uid)
    os.umask(0o077)


def load_plan(root: Path, plan_path: Path) -> list[dict[str, object]]:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read execution plan: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise PlanError("execution plan must be a v1 JSON object")
    scripts = plan.get("scripts")
    if not isinstance(scripts, list):
        raise PlanError("execution plan scripts must be a list")

    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_logs: set[Path] = set()
    for index, entry in enumerate(scripts):
        if not isinstance(entry, dict):
            raise PlanError(f"plan entry {index} is not an object")
        unit_id = str(entry.get("id", ""))
        if not UNIT_ID_RE.fullmatch(unit_id) or unit_id in seen_ids:
            raise PlanError(f"invalid or duplicate unit id: {unit_id!r}")
        if entry.get("precheck_status") != "approved":
            raise PlanError(f"{unit_id}: precheck_status must be approved")

        script = _inside(root, entry.get("path"), existing=True)
        log = _inside(root, entry.get("log_path"), existing=False)
        if not script.is_file() or script in seen_paths:
            raise PlanError(f"{unit_id}: script is missing or duplicated")
        if log in seen_logs or log == script:
            raise PlanError(f"{unit_id}: log path is duplicated or aliases the script")

        digest = str(entry.get("sha256", ""))
        if not SHA256_RE.fullmatch(digest) or _sha256(script) != digest:
            raise PlanError(f"{unit_id}: script digest does not match the approved bytes")
        timeout = entry.get("timeout_seconds")
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or not 1 <= timeout <= MAX_TIMEOUT_SECONDS
        ):
            raise PlanError(f"{unit_id}: timeout must be an integer from 1 to {MAX_TIMEOUT_SECONDS}")
        for field in ("upstream_oracle", "target_xpu_path", "xpu_proof"):
            if not str(entry.get(field, "")).strip():
                raise PlanError(f"{unit_id}: missing {field}")

        seen_ids.add(unit_id)
        seen_paths.add(script)
        seen_logs.add(log)
        normalized.append(
            {
                "id": unit_id,
                "script": script,
                "script_path": str(script.relative_to(root)),
                "log": log,
                "log_path": str(log.relative_to(root)),
                "sha256": digest,
                "timeout_seconds": timeout,
            }
        )
    return normalized


def run_plan(
    root: Path,
    python: Path,
    entries: list[dict[str, object]],
    identity: tuple[int, int, list[int]] | None = None,
) -> dict[str, object]:
    if not python.is_file():
        raise PlanError(f"python executable does not exist: {python}")
    base_environment = _safe_environment()
    results: list[dict[str, object]] = []

    for entry in entries:
        unit_id = str(entry["id"])
        script = Path(entry["script"])
        log = Path(entry["log"])
        timeout = int(entry["timeout_seconds"])
        log.parent.mkdir(parents=True, exist_ok=True)
        started_at = _utc_now()
        started = time.monotonic()
        runner_status = "completed"
        returncode: int | None = None
        error = ""
        with log.open("wb") as output, tempfile.TemporaryDirectory(
            prefix=f"xpu-alignment-{unit_id}-"
        ) as scratch:
            environment = dict(base_environment)
            preexec_fn = None
            if identity is not None:
                uid, gid, groups = identity
                os.chown(scratch, uid, gid)
                environment.update(
                    {"HOME": scratch, "TMPDIR": scratch, "USER": str(uid), "LOGNAME": str(uid)}
                )
                preexec_fn = partial(_drop_privileges, uid, gid, groups)
            output.write(
                (
                    f"XPU alignment runner\nunit_id={unit_id}\n"
                    f"script={entry['script_path']}\nsha256={entry['sha256']}\n\n"
                ).encode()
            )
            output.flush()
            # An earlier untrusted repro may have modified a later script after
            # the plan was loaded. Recheck immediately before every exec.
            if _sha256(script) != entry["sha256"]:
                runner_status = "integrity-error"
                error = "script bytes changed after plan validation"
            else:
                try:
                    completed = subprocess.run(
                        [str(python), "-I", str(script)],
                        cwd=root,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                        check=False,
                        preexec_fn=preexec_fn,
                    )
                    returncode = completed.returncode
                except subprocess.TimeoutExpired:
                    runner_status = "timeout"
                except (OSError, subprocess.SubprocessError) as exc:
                    runner_status = "launch-error"
                    error = str(exc)
            duration = time.monotonic() - started
            output.write(
                (
                    f"\n\nXPU alignment runner result\nstatus={runner_status}\n"
                    f"returncode={returncode}\nduration_seconds={duration:.3f}\n"
                ).encode()
            )

        results.append(
            {
                "id": unit_id,
                "script_path": entry["script_path"],
                "log_path": entry["log_path"],
                "sha256": entry["sha256"],
                "runner_status": runner_status,
                "timed_out": runner_status == "timeout",
                "returncode": returncode,
                "signal": -returncode if returncode is not None and returncode < 0 else None,
                "duration_seconds": round(duration, 3),
                "started_at": started_at,
                "finished_at": _utc_now(),
                "error": error or None,
            }
        )
    return {"schema_version": 1, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--user", required=True)
    args = parser.parse_args()

    try:
        root = args.root.resolve(strict=True)
        plan = (
            _inside(root, args.plan, existing=True)
            if not args.plan.is_absolute()
            else args.plan.resolve(strict=True)
        )
        results = (
            _inside(root, args.results, existing=False)
            if not args.results.is_absolute()
            else args.results.resolve()
        )
        plan.relative_to(root)
        results.relative_to(root)
        entries = load_plan(root, plan)
        payload = run_plan(root, args.python.resolve(), entries, _identity(args.user))
    except (OSError, PlanError, ValueError) as exc:
        print(f"execution plan rejected: {exc}")
        return 2

    results.parent.mkdir(parents=True, exist_ok=True)
    temporary = results.with_suffix(results.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(results)
    print(json.dumps({"executed": len(entries), "results": str(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
