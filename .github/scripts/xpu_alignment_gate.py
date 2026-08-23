#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Validate alignment artifacts and choose a deterministic publishing path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


SCHEMA_VERSION = 1
UNIT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ISSUE_TITLE_PREFIX = "[xpu-alignment]"
ISSUE_LABELS = ["ai_generated"]
EXPECTED_SOURCES = {
    "issues-created",
    "prs-created",
    "prs-merged",
    "default-branch-commits",
}
LOCAL_RESULTS = {
    "confirmed",
    "related-failure",
    "not-reproduced",
    "blocked-env",
    "blocked-platform",
    "blocked-fetch",
    "blocked-script-error",
    "needs-performance-harness",
}
ACTIONABLE_RESULTS = {"confirmed", "related-failure"}
BLOCKED_RESULTS = LOCAL_RESULTS - ACTIONABLE_RESULTS - {"not-reproduced"}
VERDICTS = {
    "needs-xpu-fix",
    "track-upstream",
    "fixed",
    "non-issue",
    "duplicate",
    "verification-gap",
}
IMPLEMENTATION_REPOSITORIES = {"intel/torch-xpu-ops", "pytorch/pytorch"}
EVENT_TYPES = {
    "issue": {"created"},
    "pr": {"created", "merged"},
    "commit": {"committed"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str, errors: list[str]) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"{label}-unreadable:{error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}-not-object")
        return {}
    return value


def _one(root: Path, pattern: str, label: str, errors: list[str]) -> Path | None:
    matches = sorted(root.rglob(pattern)) if root.is_dir() else []
    if len(matches) != 1:
        errors.append(f"{label}-count:{len(matches)}")
        return None
    return matches[0]


def _inside_file(root: Path, value: object, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}-path-invalid")
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{label}-path-outside-root")
        return None
    path = root / relative
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        errors.append(f"{label}-missing-or-outside-root")
        return None
    if not path.is_file() or path.is_symlink():
        errors.append(f"{label}-not-regular-file")
        return None
    return path


def _expected_window(scan_date: str, errors: list[str]) -> dict[str, str] | None:
    try:
        day = date.fromisoformat(scan_date)
    except ValueError:
        errors.append("gate-invalid-scan-date")
        return None
    return {
        "start": f"{day.isoformat()}T00:00:00Z",
        "end": f"{(day + timedelta(days=1)).isoformat()}T00:00:00Z",
    }


def _validate_prepare(
    root: Path, scan_date: str
) -> tuple[Path | None, dict[str, dict[str, object]], dict[str, dict[str, object]], list[str]]:
    errors: list[str] = []
    path = _one(root, "prepare.json", "prepare", errors)
    if path is None:
        return None, {}, {}, errors
    prepare = _read_json(path, "prepare", errors)
    if prepare.get("schema_version") != SCHEMA_VERSION:
        errors.append("prepare-invalid-version")
    if prepare.get("status") != "complete":
        errors.append(f"prepare-not-complete:{prepare.get('status', 'missing')}")
    blockers = prepare.get("blockers")
    if not isinstance(blockers, list) or blockers:
        errors.append("prepare-has-blockers")
    expected = _expected_window(scan_date, errors)
    if expected is not None and prepare.get("scan_window") != expected:
        errors.append("prepare-window-mismatch")
    window_start = window_end = None
    if expected is not None:
        window_start = datetime.fromisoformat(expected["start"].replace("Z", "+00:00"))
        window_end = datetime.fromisoformat(expected["end"].replace("Z", "+00:00"))

    collection = prepare.get("collection")
    queries = collection.get("queries") if isinstance(collection, dict) else None
    seen_sources: set[str] = set()
    observed_count = 0
    if not isinstance(queries, list):
        errors.append("collection-queries-not-list")
    else:
        for index, query in enumerate(queries):
            if not isinstance(query, dict):
                errors.append(f"collection-query-not-object:{index}")
                continue
            source = query.get("source")
            if source not in EXPECTED_SOURCES:
                errors.append(f"collection-query-invalid-source:{index}")
            else:
                seen_sources.add(str(source))
            if not str(query.get("request", "")).strip():
                errors.append(f"collection-query-missing-request:{index}")
            pages, count = query.get("pages"), query.get("count")
            if not isinstance(pages, int) or isinstance(pages, bool) or pages < 1:
                errors.append(f"collection-query-invalid-pages:{index}")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                errors.append(f"collection-query-invalid-count:{index}")
            else:
                observed_count += count
            if query.get("truncated") is not False:
                errors.append(f"collection-query-truncated:{index}")
            query_errors = query.get("errors")
            if not isinstance(query_errors, list) or query_errors:
                errors.append(f"collection-query-has-errors:{index}")
        if seen_sources != EXPECTED_SOURCES:
            errors.append("collection-source-coverage")
    if isinstance(collection, dict) and collection.get("observed_count") != observed_count:
        errors.append("collection-observed-count-mismatch")

    inventory: dict[str, dict[str, object]] = {}
    raw_inventory = prepare.get("inventory")
    if not isinstance(raw_inventory, list):
        errors.append("inventory-not-list")
    else:
        for index, item in enumerate(raw_inventory):
            if not isinstance(item, dict):
                errors.append(f"inventory-not-object:{index}")
                continue
            unit_id = item.get("id")
            if (
                not isinstance(unit_id, str)
                or not UNIT_ID_RE.fullmatch(unit_id)
                or unit_id in inventory
            ):
                errors.append(f"inventory-invalid-id:{unit_id}")
                continue
            kind = item.get("kind")
            if kind not in EVENT_TYPES:
                errors.append(f"inventory-invalid-kind:{unit_id}")
            if not str(item.get("url", "")).startswith("https://github.com/pytorch/pytorch/"):
                errors.append(f"inventory-invalid-url:{unit_id}")
            events = item.get("events")
            if not isinstance(events, list) or not events:
                errors.append(f"inventory-missing-events:{unit_id}")
            elif kind in EVENT_TYPES:
                event_in_window = False
                for event_index, event in enumerate(events):
                    if not isinstance(event, dict) or event.get("type") not in EVENT_TYPES[kind]:
                        errors.append(f"inventory-invalid-event:{unit_id}:{event_index}")
                        continue
                    at = event.get("at")
                    try:
                        event_time = datetime.strptime(str(at), "%Y-%m-%dT%H:%M:%SZ").replace(
                            tzinfo=timezone.utc
                        )
                    except ValueError:
                        errors.append(f"inventory-invalid-event-time:{unit_id}:{event_index}")
                        continue
                    if (
                        window_start is not None
                        and window_end is not None
                        and window_start <= event_time < window_end
                    ):
                        event_in_window = True
                if not event_in_window:
                    errors.append(f"inventory-event-outside-window:{unit_id}")
            if item.get("triage") not in {"reject", "validate"}:
                errors.append(f"inventory-invalid-triage:{unit_id}")
            if not str(item.get("reason", "")).strip():
                errors.append(f"inventory-missing-reason:{unit_id}")
            inventory[unit_id] = item
    if isinstance(collection, dict) and collection.get("unique_count") != len(inventory):
        errors.append("collection-unique-count-mismatch")

    executions: dict[str, dict[str, object]] = {}
    raw_executions = prepare.get("executions")
    if not isinstance(raw_executions, list):
        errors.append("executions-not-list")
    else:
        for entry in raw_executions:
            if not isinstance(entry, dict):
                errors.append("execution-not-object")
                continue
            unit_id = entry.get("id")
            if not isinstance(unit_id, str) or unit_id in executions:
                errors.append(f"execution-invalid-id:{unit_id}")
                continue
            if unit_id not in inventory or inventory[unit_id].get("triage") != "validate":
                errors.append(f"execution-not-validated:{unit_id}")
            script = _inside_file(root, entry.get("script"), f"script:{unit_id}", errors)
            script_digest = entry.get("script_sha256")
            if not isinstance(script_digest, str) or not SHA256_RE.fullmatch(script_digest):
                errors.append(f"execution-invalid-digest:{unit_id}")
            elif script is not None and _sha256(script) != script_digest:
                errors.append(f"execution-digest-mismatch:{unit_id}")
            timeout = entry.get("timeout_seconds")
            if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 600:
                errors.append(f"execution-invalid-timeout:{unit_id}")
            for field in ("oracle", "target_path"):
                if not str(entry.get(field, "")).strip():
                    errors.append(f"execution-missing-{field}:{unit_id}")
            executions[unit_id] = entry
    validated = {unit_id for unit_id, item in inventory.items() if item.get("triage") == "validate"}
    if set(executions) != validated:
        errors.append("execution-coverage-mismatch")
    return path, inventory, executions, errors


def _validate_runner(
    root: Path,
    prepare_path: Path | None,
    executions: dict[str, dict[str, object]],
) -> tuple[Path | None, dict[str, dict[str, object]], list[str]]:
    errors: list[str] = []
    path = _one(root, "results.json", "runner", errors)
    if path is None:
        return None, {}, errors
    runner = _read_json(path, "runner", errors)
    if runner.get("schema_version") != SCHEMA_VERSION:
        errors.append("runner-invalid-version")
    if runner.get("status") != "complete":
        errors.append(f"runner-not-complete:{runner.get('status', 'missing')}")
    if prepare_path is not None and runner.get("prepare_sha256") != _sha256(prepare_path):
        errors.append("runner-prepare-digest-mismatch")
    results: dict[str, dict[str, object]] = {}
    raw_results = runner.get("results")
    if not isinstance(raw_results, list):
        errors.append("runner-results-not-list")
    else:
        for result in raw_results:
            if not isinstance(result, dict):
                errors.append("runner-result-not-object")
                continue
            unit_id = result.get("id")
            if not isinstance(unit_id, str) or unit_id in results or unit_id not in executions:
                errors.append(f"runner-invalid-unit:{unit_id}")
                continue
            execution = executions[unit_id]
            if result.get("script_sha256") != execution.get("script_sha256"):
                errors.append(f"runner-script-digest-mismatch:{unit_id}")
            log = _inside_file(root, result.get("log"), f"runner-log:{unit_id}", errors)
            log_digest = result.get("log_sha256")
            if not isinstance(log_digest, str) or not SHA256_RE.fullmatch(log_digest):
                errors.append(f"runner-invalid-log-digest:{unit_id}")
            elif log is not None and _sha256(log) != log_digest:
                errors.append(f"runner-log-digest-mismatch:{unit_id}")
            if not isinstance(result.get("command"), list) or not result["command"]:
                errors.append(f"runner-invalid-command:{unit_id}")
            if not isinstance(result.get("timed_out"), bool):
                errors.append(f"runner-invalid-timeout-state:{unit_id}")
            returncode = result.get("returncode")
            if returncode is not None and (
                not isinstance(returncode, int) or isinstance(returncode, bool)
            ):
                errors.append(f"runner-invalid-returncode:{unit_id}")
            error = result.get("error")
            if error is not None and not isinstance(error, str):
                errors.append(f"runner-invalid-error:{unit_id}")
            results[unit_id] = result
    if set(results) != set(executions):
        errors.append("runner-coverage-mismatch")
    return path, results, errors


def _validate_scan(
    root: Path,
    runner_root: Path,
    prepare_path: Path | None,
    runner_path: Path | None,
    executions: dict[str, dict[str, object]],
    results: dict[str, dict[str, object]],
) -> tuple[Path | None, list[str], list[str]]:
    errors: list[str] = []
    path = _one(root, "scan.json", "scan", errors)
    if path is None:
        return None, [], errors
    scan = _read_json(path, "scan", errors)
    if scan.get("schema_version") != SCHEMA_VERSION:
        errors.append("scan-invalid-version")
    if prepare_path is not None and scan.get("prepare_sha256") != _sha256(prepare_path):
        errors.append("scan-prepare-digest-mismatch")
    if runner_path is not None and scan.get("runner_sha256") != _sha256(runner_path):
        errors.append("scan-runner-digest-mismatch")
    if scan.get("status") != "complete":
        errors.append(f"scan-not-complete:{scan.get('status', 'missing')}")
    blockers = scan.get("blockers")
    if not isinstance(blockers, list) or blockers:
        errors.append("scan-has-blockers")
    candidates: dict[str, dict[str, object]] = {}
    raw_candidates = scan.get("candidates")
    if not isinstance(raw_candidates, list):
        errors.append("scan-candidates-not-list")
    else:
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                errors.append("scan-candidate-not-object")
                continue
            unit_id = candidate.get("id")
            if not isinstance(unit_id, str) or unit_id in candidates or unit_id not in executions:
                errors.append(f"scan-invalid-unit:{unit_id}")
                continue
            result = candidate.get("local_result")
            if result not in LOCAL_RESULTS:
                errors.append(f"scan-invalid-result:{unit_id}")
            if result in BLOCKED_RESULTS:
                errors.append(f"scan-blocked-result:{unit_id}:{result}")
            runner_result = results.get(unit_id, {})
            if result in ACTIONABLE_RESULTS | {"not-reproduced"}:
                if runner_result.get("timed_out") or runner_result.get("error") is not None:
                    errors.append(f"scan-result-contradicts-runner:{unit_id}")
                if candidate.get("target_path_verified") is not True:
                    errors.append(f"scan-target-unverified:{unit_id}")
            evidence = _inside_file(
                runner_root,
                candidate.get("evidence"),
                f"scan-evidence:{unit_id}",
                errors,
            )
            expected_log = runner_result.get("log")
            if evidence is not None and candidate.get("evidence") != expected_log:
                errors.append(f"scan-evidence-mismatch:{unit_id}")
            candidates[unit_id] = candidate
    if set(candidates) != set(executions):
        errors.append("scan-coverage-mismatch")
    actionable = sorted(
        unit_id
        for unit_id, candidate in candidates.items()
        if candidate.get("local_result") in ACTIONABLE_RESULTS
    )
    return path, actionable, errors


def _validate_review(
    root: Path, scan_path: Path | None, actionable: list[str]
) -> tuple[list[dict[str, object]], dict[str, str], list[str]]:
    errors: list[str] = []
    path = _one(root, "review.json", "review", errors)
    if path is None:
        return [], {}, errors
    review = _read_json(path, "review", errors)
    if review.get("schema_version") != SCHEMA_VERSION:
        errors.append("review-invalid-version")
    if review.get("status") != "complete":
        errors.append(f"review-not-complete:{review.get('status', 'missing')}")
    blockers = review.get("blockers")
    if not isinstance(blockers, list) or blockers:
        errors.append("review-has-blockers")
    if scan_path is not None and review.get("scan_sha256") != _sha256(scan_path):
        errors.append("review-scan-digest-mismatch")
    units = review.get("units")
    if not isinstance(units, list):
        errors.append("review-units-not-list")
        return [], {}, errors
    verdicts: dict[str, str] = {}
    payloads: list[dict[str, object]] = []
    for entry in units:
        if not isinstance(entry, dict):
            errors.append("review-unit-not-object")
            continue
        unit_id, verdict = entry.get("id"), entry.get("verdict")
        if not isinstance(unit_id, str) or unit_id in verdicts or unit_id not in actionable:
            errors.append(f"review-invalid-unit:{unit_id}")
            continue
        if verdict not in VERDICTS:
            errors.append(f"review-invalid-verdict:{unit_id}")
            continue
        verdicts[unit_id] = str(verdict)
        if entry.get("implementation_repository") not in IMPLEMENTATION_REPOSITORIES:
            errors.append(f"review-invalid-repository:{unit_id}")
        tracker = entry.get("canonical_tracker")
        if tracker is not None and (
            not isinstance(tracker, str)
            or not tracker.startswith("https://github.com/intel/torch-xpu-ops/issues/")
        ):
            errors.append(f"review-invalid-tracker:{unit_id}")
        payload = entry.get("payload")
        expects_payload = verdict == "needs-xpu-fix" and tracker is None
        if not expects_payload:
            if payload is not None:
                errors.append(f"review-unexpected-payload:{unit_id}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"review-missing-payload:{unit_id}")
            continue
        title, body = payload.get("title"), payload.get("body")
        if not isinstance(title, str) or not title.startswith(f"{ISSUE_TITLE_PREFIX} "):
            errors.append(f"payload-invalid-title:{unit_id}")
        if not isinstance(body, str) or not body.strip():
            errors.append(f"payload-empty-body:{unit_id}")
        if payload.get("labels") != ISSUE_LABELS:
            errors.append(f"payload-invalid-labels:{unit_id}")
        payloads.append({"unit_id": unit_id, "title": title, "body": body, "labels": ISSUE_LABELS})
    if sorted(verdicts) != actionable:
        errors.append("review-coverage-mismatch")
    return payloads, dict(sorted(verdicts.items())), errors


def build_decision(
    prepare_root: Path,
    runner_root: Path,
    scan_root: Path,
    review_root: Path,
    *,
    mode: str,
    producers_clean: bool,
    run_id: str,
    scan_date: str,
) -> dict[str, object]:
    if mode not in {"schedule", "dry-run"}:
        raise ValueError(f"unsupported mode: {mode}")
    prepare_path, _, executions, prepare_errors = _validate_prepare(prepare_root, scan_date)
    runner_path, results, runner_errors = _validate_runner(runner_root, prepare_path, executions)
    scan_path, actionable, scan_errors = _validate_scan(
        scan_root, runner_root, prepare_path, runner_path, executions, results
    )
    payloads, verdicts, review_errors = _validate_review(review_root, scan_path, actionable)
    blockers = prepare_errors + runner_errors + scan_errors + review_errors
    if not producers_clean:
        blockers.append("producer-job-failed")
    attention_reasons = []
    if "verification-gap" in verdicts.values():
        attention_reasons.append("review-verification-gap")

    if blockers:
        would_decision = "blocked"
        published_payloads: list[dict[str, object]] = []
    elif not payloads:
        would_decision = "none"
        published_payloads = []
    elif len(payloads) == 1:
        would_decision = "file-one"
        published_payloads = payloads
    else:
        would_decision = "triage"
        published_payloads = payloads
    decision = "dry-run" if mode == "dry-run" else would_decision
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scan_date": scan_date,
        "mode": mode,
        "decision": decision,
        "would_decision": would_decision,
        "needs_attention": bool(blockers or attention_reasons),
        "attention_reasons": attention_reasons,
        "blockers": blockers,
        "mandatory_units": actionable,
        "unit_verdicts": verdicts,
        "actionable_units": sorted(str(item["unit_id"]) for item in published_payloads),
        "payloads": published_payloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--scan-date", required=True)
    parser.add_argument("--mode", choices=("schedule", "dry-run"), required=True)
    parser.add_argument("--producers-clean", action="store_true")
    args = parser.parse_args()
    decision = build_decision(
        args.prepare_root,
        args.runner_root,
        args.scan_root,
        args.review_root,
        mode=args.mode,
        producers_clean=args.producers_clean,
        run_id=args.run_id,
        scan_date=args.scan_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**decision, "payloads": decision["actionable_units"]}, indent=2))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"decision={decision['decision']}\n")
            handle.write(f"needs_attention={'true' if decision['needs_attention'] else 'false'}\n")
            handle.write(f"actionable_count={len(decision['actionable_units'])}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
