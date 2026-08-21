#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Validate XPU alignment v1 artifacts and decide what may be published."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from alignment_triage import ISSUE_LABELS, ISSUE_TITLE_PREFIX, UNIT_ID_RE


SCHEMA_VERSION = 1
ACTIONABLE_RESULTS = frozenset({"confirmed", "related-failure"})
LOCAL_RESULTS = ACTIONABLE_RESULTS | frozenset(
    {
        "not-reproduced",
        "blocked-env",
        "blocked-platform",
        "blocked-fetch",
        "blocked-script-error",
        "needs-performance-harness",
    }
)
REJECTION_CATEGORIES = frozenset(
    {
        "docs-ci-release",
        "platform-exclusive",
        "test-toggle",
        "nonfunctional",
        "duplicate-chain",
        "insufficient-repro-context",
        "nonbug",
        "no-shared-bug-signal",
        "other",
    }
)
ALLOWED_VERDICTS = frozenset(
    {
        "needs-xpu-fix",
        "track-upstream",
        "fixed",
        "non-issue",
        "duplicate",
        "verification-gap",
    }
)
IMPLEMENTATION_REPOSITORIES = frozenset(
    {"intel/torch-xpu-ops", "pytorch/pytorch", "none", "unresolved"}
)
SCAN_MANIFEST_GLOB = "**/scan_manifest.json"
REVIEW_MANIFEST_GLOB = "**/review/review_manifest.json"


def _read_json(path: Path, label: str, errors: list[str]) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}-unreadable:{path}:{exc}")
        return None


def _timestamp(value: object, label: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        errors.append(f"{label}-invalid-timestamp")
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{label}-invalid-timestamp")
        return None
    if parsed.tzinfo != timezone.utc:
        errors.append(f"{label}-not-utc")
        return None
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_path(root: Path, pattern: str, label: str, errors: list[str]) -> Path | None:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        errors.append(f"{label}-count:{len(paths)}")
        return None
    return paths[0]


def _relative_child(run: Path, value: object, label: str, errors: list[str]) -> Path | None:
    relative = Path(str(value or ""))
    if not str(value or "") or relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{label}-unsafe-path:{value}")
        return None
    path = (run / relative).resolve()
    try:
        path.relative_to(run.resolve())
    except ValueError:
        errors.append(f"{label}-unsafe-path:{value}")
        return None
    return path


def load_scan(root: Path) -> tuple[dict[str, object], Path | None, list[str]]:
    errors: list[str] = []
    path = _one_path(root, SCAN_MANIFEST_GLOB, "scan-manifest", errors)
    if path is None:
        return {}, None, errors
    manifest = _read_json(path, "scan-manifest", errors)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("scan-manifest-invalid-version")
        return {}, path.parent, errors
    if manifest.get("mode") != "automation":
        errors.append("scan-manifest-invalid-mode")
    if manifest.get("phase") not in {"prepared", "final"}:
        errors.append("scan-manifest-invalid-phase")
    if manifest.get("status") not in {"complete", "incomplete", "blocked"}:
        errors.append("scan-manifest-invalid-status")
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list):
        errors.append("scan-manifest-blockers-not-list")
    elif manifest.get("status") == "complete" and blockers:
        errors.append("scan-manifest-complete-with-blockers")

    window = manifest.get("scan_window")
    if not isinstance(window, dict) or window.get("timezone") != "UTC":
        errors.append("scan-manifest-invalid-window")
    else:
        start = _timestamp(window.get("start"), "scan-window-start", errors)
        end = _timestamp(window.get("end"), "scan-window-end", errors)
        if start is not None and end is not None and start >= end:
            errors.append("scan-manifest-empty-window")
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        errors.append("scan-manifest-invalid-environment")
    else:
        for field in ("python", "torch", "device"):
            if not str(environment.get(field, "")).strip():
                errors.append(f"scan-environment-missing:{field}")
        if not isinstance(environment.get("xpu_available"), bool):
            errors.append("scan-environment-invalid:xpu_available")
        elif manifest.get("status") == "complete" and not environment["xpu_available"]:
            errors.append("scan-complete-without-xpu")

    collection = manifest.get("collection")
    if not isinstance(collection, dict) or collection.get("status") not in {
        "complete",
        "incomplete",
        "blocked",
    }:
        errors.append("scan-manifest-invalid-collection")
    else:
        collection_errors = collection.get("errors")
        if not isinstance(collection_errors, list):
            errors.append("scan-collection-errors-not-list")
        elif collection.get("status") == "complete" and collection_errors:
            errors.append("scan-collection-complete-with-errors")
        sources = collection.get("sources")
        if not isinstance(sources, dict):
            errors.append("scan-manifest-missing-sources")
        else:
            expected_events = {
                "issues": ["created"],
                "prs": ["created", "merged"],
                "commits": ["default-branch"],
            }
            for source, event_types in expected_events.items():
                entry = sources.get(source)
                if not isinstance(entry, dict):
                    errors.append(f"scan-source-missing:{source}")
                    continue
                if entry.get("event_types") != event_types:
                    errors.append(f"scan-source-events:{source}")
                queries = entry.get("queries")
                if not isinstance(queries, list) or not queries:
                    errors.append(f"scan-source-queries:{source}")
                    queries = []
                query_pages = 0
                query_truncated = False
                for query_number, query in enumerate(queries, start=1):
                    prefix = f"scan-source-query:{source}:{query_number}"
                    if not isinstance(query, dict):
                        errors.append(f"{prefix}:not-object")
                        continue
                    if not str(query.get("request", "")).strip():
                        errors.append(f"{prefix}:request")
                    pages = query.get("pages")
                    if not isinstance(pages, int) or isinstance(pages, bool) or pages < 0:
                        errors.append(f"{prefix}:pages")
                    else:
                        query_pages += pages
                        if collection.get("status") == "complete" and pages < 1:
                            errors.append(f"{prefix}:complete-without-request")
                    count = query.get("count")
                    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                        errors.append(f"{prefix}:count")
                    truncated = query.get("truncated")
                    if not isinstance(truncated, bool):
                        errors.append(f"{prefix}:truncated")
                    elif truncated:
                        query_truncated = True
                for field in ("pages", "count"):
                    value = entry.get(field)
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        errors.append(f"scan-source-{field}:{source}")
                if isinstance(entry.get("pages"), int) and not isinstance(
                    entry.get("pages"), bool
                ) and entry.get("pages") != query_pages:
                    errors.append(f"scan-source-page-total:{source}")
                if not isinstance(entry.get("truncated"), bool):
                    errors.append(f"scan-source-truncated:{source}")
                elif entry.get("truncated") != query_truncated:
                    errors.append(f"scan-source-truncation-summary:{source}")
                if collection.get("status") == "complete" and query_truncated:
                    errors.append(f"scan-source-complete-but-truncated:{source}")
    if manifest.get("phase") == "final" and not (path.parent / "reports/scan_report.md").is_file():
        errors.append("scan-report-missing")
    return manifest, path.parent, errors


def load_ledger(
    run: Path | None, manifest: dict[str, object]
) -> tuple[dict[str, dict[str, object]], list[str]]:
    errors: list[str] = []
    if run is None:
        return {}, ["ledger-run-missing"]
    path = _relative_child(run, manifest.get("ledger"), "ledger", errors)
    if path is None:
        return {}, errors
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, errors + [f"ledger-unreadable:{path}:{exc}"]

    rows: dict[str, dict[str, object]] = {}
    window = manifest.get("scan_window")
    window_errors: list[str] = []
    start = _timestamp(window.get("start"), "ledger-window-start", window_errors) if isinstance(window, dict) else None
    end = _timestamp(window.get("end"), "ledger-window-end", window_errors) if isinstance(window, dict) else None
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"ledger-unparsable:{number}")
            continue
        if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"ledger-invalid-row:{number}")
            continue
        unit_id = str(row.get("id", ""))
        if not UNIT_ID_RE.fullmatch(unit_id) or unit_id in rows:
            errors.append(f"ledger-invalid-or-duplicate-id:{unit_id or number}")
            continue
        kind = row.get("kind")
        if kind not in {"issue", "pr", "commit"}:
            errors.append(f"ledger-invalid-kind:{unit_id}")
        if not str(row.get("title", "")).strip() or not str(row.get("url", "")).startswith(
            "https://github.com/pytorch/pytorch/"
        ):
            errors.append(f"ledger-missing-source:{unit_id}")
        events = row.get("events")
        if not isinstance(events, list) or not events:
            errors.append(f"ledger-missing-events:{unit_id}")
        else:
            allowed_events = {
                "issue": {"created"},
                "pr": {"created", "merged"},
                "commit": {"default-branch"},
            }.get(str(kind), set())
            for event in events:
                if not isinstance(event, dict) or event.get("type") not in allowed_events:
                    errors.append(f"ledger-invalid-event:{unit_id}")
                    continue
                observed = _timestamp(event.get("at"), f"ledger-event:{unit_id}", errors)
                if observed is not None and start is not None and end is not None:
                    if not start <= observed < end:
                        errors.append(f"ledger-event-outside-window:{unit_id}")

        triage = row.get("triage_status")
        validation = row.get("validation_status")
        result = row.get("local_result")
        if triage not in {"pending", "reject", "validate"}:
            errors.append(f"ledger-invalid-triage:{unit_id}")
        elif triage == "reject":
            if validation != "not-needed" or result is not None:
                errors.append(f"ledger-invalid-rejection-state:{unit_id}")
            if not str(row.get("triage_reason", "")).strip():
                errors.append(f"ledger-missing-rejection-reason:{unit_id}")
            category = row.get("rejection_category")
            if category not in REJECTION_CATEGORIES:
                errors.append(f"ledger-invalid-rejection-category:{unit_id}")
            if category == "duplicate-chain":
                duplicate = str(row.get("duplicate_of", ""))
                if not UNIT_ID_RE.fullmatch(duplicate) or duplicate == unit_id:
                    errors.append(f"ledger-invalid-duplicate:{unit_id}")
        elif triage == "validate":
            if validation not in {"pending", "done"}:
                errors.append(f"ledger-invalid-validation:{unit_id}")
            elif validation == "pending" and result is not None:
                errors.append(f"ledger-result-before-validation:{unit_id}")
            elif validation == "done" and result not in LOCAL_RESULTS:
                errors.append(f"ledger-invalid-result:{unit_id}")
            if row.get("rejection_category") is not None:
                errors.append(f"ledger-category-on-selected:{unit_id}")
        rows[unit_id] = row

    for unit_id, row in rows.items():
        duplicate = row.get("duplicate_of")
        if duplicate is not None and duplicate not in rows:
            errors.append(f"ledger-duplicate-target-missing:{unit_id}")
    return rows, window_errors + errors


def validate_raw_candidates(
    run: Path | None,
    manifest: dict[str, object],
    rows: dict[str, dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    if run is None:
        return ["raw-candidates-run-missing"]
    path = _relative_child(run, manifest.get("raw_candidates"), "raw-candidates", errors)
    if path is None:
        return errors
    raw = _read_json(path, "raw-candidates", errors)
    if not isinstance(raw, list):
        return errors + ["raw-candidates-not-list"]

    objects: dict[str, dict[str, object]] = {}
    kind_counts = {"issue": 0, "pr": 0, "commit": 0}
    for entry in raw:
        if not isinstance(entry, dict):
            errors.append("raw-candidate-not-object")
            continue
        unit_id = str(entry.get("id", ""))
        kind = entry.get("kind")
        if not UNIT_ID_RE.fullmatch(unit_id) or unit_id in objects:
            errors.append(f"raw-candidate-invalid-or-duplicate-id:{unit_id}")
            continue
        if kind not in kind_counts:
            errors.append(f"raw-candidate-invalid-kind:{unit_id}")
        else:
            kind_counts[str(kind)] += 1
        objects[unit_id] = entry

    if set(objects) != set(rows):
        errors.append("raw-candidate-ledger-id-mismatch")
    for unit_id in set(objects) & set(rows):
        raw_entry = objects[unit_id]
        ledger_entry = rows[unit_id]
        if any(
            raw_entry.get(field) != ledger_entry.get(field)
            for field in ("kind", "title", "url", "events")
        ):
            errors.append(f"raw-candidate-ledger-source-mismatch:{unit_id}")

    collection = manifest.get("collection")
    sources = collection.get("sources") if isinstance(collection, dict) else None
    if isinstance(sources, dict):
        for kind, source in {"issue": "issues", "pr": "prs", "commit": "commits"}.items():
            source_entry = sources.get(source)
            if isinstance(source_entry, dict) and source_entry.get("count") != kind_counts[kind]:
                errors.append(f"raw-candidate-source-count-mismatch:{source}")
    return errors


def validate_execution(
    run: Path | None,
    manifest: dict[str, object],
    rows: dict[str, dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    if run is None:
        return ["execution-run-missing"]
    plan_path = _relative_child(run, manifest.get("execution_plan"), "execution-plan", errors)
    results_path = _relative_child(
        run, manifest.get("execution_results"), "execution-results", errors
    )
    if plan_path is None or results_path is None:
        return errors
    plan = _read_json(plan_path, "execution-plan", errors)
    results = _read_json(results_path, "execution-results", errors)
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA_VERSION:
        errors.append("execution-plan-invalid-version")
        return errors
    if not isinstance(results, dict) or results.get("schema_version") != SCHEMA_VERSION:
        errors.append("execution-results-invalid-version")
        return errors

    plan_entries: dict[str, dict[str, object]] = {}
    scripts = plan.get("scripts")
    if not isinstance(scripts, list):
        errors.append("execution-plan-scripts-not-list")
        scripts = []
    for entry in scripts:
        if not isinstance(entry, dict):
            errors.append("execution-plan-malformed-entry")
            continue
        unit_id = str(entry.get("id", ""))
        if unit_id not in rows or unit_id in plan_entries:
            errors.append(f"execution-plan-invalid-unit:{unit_id}")
            continue
        if rows[unit_id].get("triage_status") != "validate":
            errors.append(f"execution-plan-unselected-unit:{unit_id}")
        if entry.get("precheck_status") != "approved":
            errors.append(f"execution-plan-unapproved:{unit_id}")
        timeout = entry.get("timeout_seconds")
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or not 1 <= timeout <= 600
        ):
            errors.append(f"execution-plan-invalid-timeout:{unit_id}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"execution-plan-invalid-digest:{unit_id}")
        for field in ("upstream_oracle", "target_xpu_path", "xpu_proof"):
            if not str(entry.get(field, "")).strip():
                errors.append(f"execution-plan-missing-{field}:{unit_id}")
        script = _relative_child(run, entry.get("path"), f"execution-script:{unit_id}", errors)
        if script is None or not script.is_file():
            errors.append(f"execution-script-missing:{unit_id}")
        elif _sha256(script) != entry.get("sha256"):
            errors.append(f"execution-script-digest:{unit_id}")
        _relative_child(run, entry.get("log_path"), f"execution-planned-log:{unit_id}", errors)
        plan_entries[unit_id] = entry

    result_entries: dict[str, dict[str, object]] = {}
    raw_results = results.get("results")
    if not isinstance(raw_results, list):
        errors.append("execution-results-not-list")
        raw_results = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            errors.append("execution-result-malformed-entry")
            continue
        unit_id = str(entry.get("id", ""))
        if unit_id not in plan_entries or unit_id in result_entries:
            errors.append(f"execution-result-invalid-unit:{unit_id}")
            continue
        plan_entry = plan_entries[unit_id]
        if (
            entry.get("script_path") != plan_entry.get("path")
            or entry.get("log_path") != plan_entry.get("log_path")
            or entry.get("sha256") != plan_entry.get("sha256")
        ):
            errors.append(f"execution-result-plan-mismatch:{unit_id}")
        runner_status = entry.get("runner_status")
        if runner_status not in {
            "completed",
            "timeout",
            "launch-error",
            "integrity-error",
        }:
            errors.append(f"execution-result-invalid-status:{unit_id}")
        if runner_status == "completed" and (
            not isinstance(entry.get("returncode"), int)
            or isinstance(entry.get("returncode"), bool)
            or entry.get("timed_out") is not False
        ):
            errors.append(f"execution-result-invalid-completion:{unit_id}")
        if runner_status == "timeout" and (
            entry.get("returncode") is not None or entry.get("timed_out") is not True
        ):
            errors.append(f"execution-result-invalid-timeout:{unit_id}")
        duration = entry.get("duration_seconds")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or duration < 0
        ):
            errors.append(f"execution-result-invalid-duration:{unit_id}")
        _timestamp(entry.get("started_at"), f"execution-start:{unit_id}", errors)
        _timestamp(entry.get("finished_at"), f"execution-finish:{unit_id}", errors)
        log = _relative_child(run, entry.get("log_path"), f"execution-log:{unit_id}", errors)
        if log is None or not log.is_file():
            errors.append(f"execution-log-missing:{unit_id}")
        result_entries[unit_id] = entry
    if set(result_entries) != set(plan_entries):
        errors.append("execution-result-coverage-mismatch")

    evidence_results = ACTIONABLE_RESULTS | {"not-reproduced"}
    for unit_id, row in rows.items():
        if row.get("validation_status") != "done" or row.get("local_result") not in evidence_results:
            continue
        if unit_id not in plan_entries or unit_id not in result_entries:
            errors.append(f"execution-evidence-missing:{unit_id}")
            continue
        if row.get("repro_path") != plan_entries[unit_id].get("path"):
            errors.append(f"execution-ledger-repro-mismatch:{unit_id}")
        if row.get("log_path") != result_entries[unit_id].get("log_path"):
            errors.append(f"execution-ledger-log-mismatch:{unit_id}")
    return errors


def negative_categories(rows: dict[str, dict[str, object]]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = defaultdict(list)
    for unit_id, row in rows.items():
        if row.get("triage_status") == "reject":
            categories[str(row["rejection_category"])].append(unit_id)
        elif row.get("validation_status") == "done" and row.get("local_result") not in ACTIONABLE_RESULTS:
            categories[str(row["local_result"])].append(unit_id)
    return {category: sorted(ids) for category, ids in categories.items()}


def load_review(
    root: Path,
    scan_run: Path,
    expected_review_run: Path,
    rows: dict[str, dict[str, object]],
) -> tuple[dict[str, str], list[str], list[str], list[dict[str, object]], list[str]]:
    errors: list[str] = []
    path = _one_path(root, REVIEW_MANIFEST_GLOB, "review-manifest", errors)
    if path is None:
        return {}, [], [], [], errors
    if path.resolve() != (expected_review_run / "review/review_manifest.json").resolve():
        errors.append("review-run-mismatch")
    manifest = _read_json(path, "review-manifest", errors)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("review-manifest-invalid-version")
        return {}, [], [], [], errors
    if manifest.get("review_status") != "complete":
        errors.append(f"review-not-complete:{manifest.get('review_status')}")
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list) or blockers:
        errors.append("review-has-blockers")
    report = path.parent / "review_report.md"
    if not report.is_file():
        errors.append("review-report-missing")

    mandatory = sorted(
        unit_id
        for unit_id, row in rows.items()
        if row.get("validation_status") == "done" and row.get("local_result") in ACTIONABLE_RESULTS
    )
    declared_mandatory = manifest.get("mandatory_units")
    if (
        not isinstance(declared_mandatory, list)
        or any(not isinstance(unit_id, str) for unit_id in declared_mandatory)
        or sorted(declared_mandatory) != mandatory
    ):
        errors.append("review-mandatory-set-mismatch")

    policy = manifest.get("sample_policy")
    if policy != {"per_category": 3, "order": "id-lexical"}:
        errors.append("review-invalid-sample-policy")
    sample_count = 3

    expected_samples = {
        (unit_id, category)
        for category, ids in negative_categories(rows).items()
        for unit_id in ids[:sample_count]
    }
    declared_samples: set[tuple[str, str]] = set()
    samples = manifest.get("negative_samples")
    if not isinstance(samples, list):
        errors.append("review-negative-samples-not-list")
    else:
        for entry in samples:
            if not isinstance(entry, dict):
                errors.append("review-malformed-negative-sample")
                continue
            pair = (str(entry.get("id", "")), str(entry.get("category", "")))
            if pair in declared_samples or entry.get("outcome") not in {"accepted", "promoted"}:
                errors.append(f"review-invalid-negative-sample:{pair[0]}")
            declared_samples.add(pair)
    if declared_samples != expected_samples:
        errors.append("review-negative-sample-mismatch")

    promoted = manifest.get("promoted_units")
    if (
        not isinstance(promoted, list)
        or any(
            not isinstance(unit_id, str) or unit_id not in rows or unit_id in mandatory
            for unit_id in promoted
        )
        or len(set(promoted)) != len(promoted)
    ):
        errors.append("review-invalid-promotions")
        promoted_ids: list[str] = []
    else:
        promoted_ids = promoted
        sample_outcomes: dict[str, object] = {}
        if isinstance(samples, list):
            sample_outcomes = {
                str(entry.get("id", "")): entry.get("outcome")
                for entry in samples
                if isinstance(entry, dict)
            }
        if any(sample_outcomes.get(unit_id) != "promoted" for unit_id in promoted_ids):
            errors.append("review-promotion-not-sampled")

    expected_units = set(mandatory) | set(promoted_ids)
    verdicts: dict[str, str] = {}
    payload_paths: dict[str, str] = {}
    needs_fix: list[str] = []
    units = manifest.get("units")
    if not isinstance(units, list):
        errors.append("review-units-not-list")
        units = []
    for entry in units:
        if not isinstance(entry, dict):
            errors.append("review-malformed-unit")
            continue
        unit_id = str(entry.get("id", ""))
        verdict = str(entry.get("verdict", ""))
        if unit_id not in expected_units or unit_id in verdicts or verdict not in ALLOWED_VERDICTS:
            errors.append(f"review-invalid-unit:{unit_id or '<unnamed>'}")
            continue
        if entry.get("implementation_repository") not in IMPLEMENTATION_REPOSITORIES:
            errors.append(f"review-invalid-implementation-repository:{unit_id}")
        canonical = entry.get("canonical_tracker")
        if canonical is not None and (
            not isinstance(canonical, str)
            or not canonical.startswith("https://github.com/intel/torch-xpu-ops/issues/")
        ):
            errors.append(f"review-invalid-canonical-tracker:{unit_id}")
        payload = entry.get("payload")
        if verdict == "needs-xpu-fix":
            needs_fix.append(unit_id)
            if canonical is None:
                expected_payload = f"review/final_issue_{unit_id}.json"
                if payload != expected_payload:
                    errors.append(f"review-missing-payload:{unit_id}")
                else:
                    payload_paths[unit_id] = payload
            elif payload is not None:
                errors.append(f"review-payload-for-existing-tracker:{unit_id}")
        elif payload is not None:
            errors.append(f"review-payload-for-nonactionable:{unit_id}")
        verdicts[unit_id] = verdict
    if set(verdicts) != expected_units:
        errors.append("review-unit-coverage-mismatch")

    payloads: list[dict[str, object]] = []
    declared_files: set[Path] = set()
    review_run = path.parent.parent
    for unit_id, relative in payload_paths.items():
        payload_path = _relative_child(review_run, relative, f"payload:{unit_id}", errors)
        if payload_path is None:
            continue
        declared_files.add(payload_path)
        scan_copy = scan_run / payload_path.relative_to(review_run)
        if scan_copy.exists():
            errors.append(f"payload-authored-before-review:{unit_id}")
            continue
        payload = _read_json(payload_path, f"payload:{unit_id}", errors)
        if not isinstance(payload, dict):
            continue
        if payload.get("unit_id") != unit_id:
            errors.append(f"payload-unit-mismatch:{unit_id}")
        title = str(payload.get("title", ""))
        if not title.startswith(f"{ISSUE_TITLE_PREFIX} "):
            errors.append(f"payload-invalid-title:{unit_id}")
        if not str(payload.get("body", "")).strip():
            errors.append(f"payload-empty-body:{unit_id}")
        if payload.get("labels") != ISSUE_LABELS:
            errors.append(f"payload-invalid-labels:{unit_id}")
        payloads.append(
            {
                "unit_id": unit_id,
                "title": title,
                "body": str(payload.get("body", "")),
                "labels": ISSUE_LABELS,
            }
        )
    actual_files = {candidate.resolve() for candidate in path.parent.glob("final_issue_*.json")}
    if actual_files != declared_files:
        errors.append("review-payload-file-set-mismatch")
    return verdicts, sorted(needs_fix), promoted_ids, payloads, errors


def build_decision(
    root: Path,
    *,
    auto_file: bool,
    run_id: str = "",
    scan_date: str = "",
    scan_root: Path | None = None,
) -> dict[str, object]:
    immutable_scan_root = scan_root or root
    scan, run, scan_errors = load_scan(immutable_scan_root)
    rows, ledger_errors = load_ledger(run, scan)
    raw_errors = validate_raw_candidates(run, scan, rows)
    execution_errors = validate_execution(run, scan, rows)
    if run is None:
        expected_review_run = root
    else:
        try:
            expected_review_run = root / run.relative_to(immutable_scan_root)
        except ValueError:
            expected_review_run = root
            scan_errors.append("scan-run-outside-root")
    verdicts, needs_fix, promoted, payloads, review_errors = load_review(
        root, run or immutable_scan_root, expected_review_run, rows
    )

    if scan_date:
        try:
            expected_start = f"{date.fromisoformat(scan_date).isoformat()}T00:00:00Z"
            expected_end = f"{(date.fromisoformat(scan_date) + timedelta(days=1)).isoformat()}T00:00:00Z"
        except ValueError:
            scan_errors.append("gate-invalid-scan-date")
        else:
            window = scan.get("scan_window")
            if not isinstance(window, dict) or (
                window.get("start") != expected_start or window.get("end") != expected_end
            ):
                scan_errors.append("scan-window-does-not-match-gate-date")

    pending = sorted(
        unit_id
        for unit_id, row in rows.items()
        if row.get("triage_status") == "pending"
        or (
            row.get("triage_status") == "validate"
            and row.get("validation_status") == "pending"
        )
    )
    collection = scan.get("collection") if isinstance(scan.get("collection"), dict) else {}
    sources = collection.get("sources") if isinstance(collection, dict) else {}
    enumeration_complete = (
        isinstance(collection, dict)
        and collection.get("status") == "complete"
        and isinstance(sources, dict)
        and all(
            isinstance(sources.get(name), dict) and sources[name].get("truncated") is False
            for name in ("issues", "prs", "commits")
        )
    )
    scan_complete = (
        scan.get("phase") == "final"
        and scan.get("status") == "complete"
        and enumeration_complete
        and not pending
    )
    blockers = scan_errors + ledger_errors + raw_errors + execution_errors + review_errors
    attention_reasons: list[str] = []
    if not enumeration_complete:
        attention_reasons.append("collection-incomplete")
    if scan.get("phase") != "final":
        attention_reasons.append("scan-not-final")
    if scan.get("status") != "complete":
        attention_reasons.append(f"scan-{scan.get('status', 'missing')}")
    if pending:
        attention_reasons.append("pending-validation")
    unresolved = sorted(
        unit_id
        for unit_id, row in rows.items()
        if row.get("validation_status") == "done"
        and row.get("local_result") not in ACTIONABLE_RESULTS | {"not-reproduced"}
    )
    if unresolved:
        attention_reasons.append("unresolved-validation")
    review_unresolved = sorted(
        unit_id for unit_id, verdict in verdicts.items() if verdict == "verification-gap"
    )
    if review_unresolved:
        attention_reasons.append("review-verification-gap")
    if promoted:
        attention_reasons.append("negative-sample-promotion")

    publishable = sorted(str(payload["unit_id"]) for payload in payloads)
    if blockers:
        decision = "blocked"
        decision_payloads: list[dict[str, object]] = []
    elif not publishable:
        decision = "none"
        decision_payloads = []
    elif (
        scan_complete
        and not unresolved
        and not review_unresolved
        and not promoted
        and auto_file
        and len(publishable) == 1
    ):
        decision = "file-one"
        decision_payloads = payloads
    else:
        decision = "triage"
        decision_payloads = payloads

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scan_date": scan_date,
        "decision": decision,
        "needs_attention": bool(blockers or attention_reasons),
        "scan_complete": scan_complete,
        "collection_complete": enumeration_complete,
        "pending_units": pending,
        "unresolved_units": unresolved,
        "review_unresolved_units": review_unresolved,
        "mandatory_units": sorted(
            unit_id
            for unit_id, row in rows.items()
            if row.get("validation_status") == "done"
            and row.get("local_result") in ACTIONABLE_RESULTS
        ),
        "promoted_units": sorted(promoted),
        "unit_verdicts": dict(sorted(verdicts.items())),
        "needs_xpu_fix_units": sorted(needs_fix),
        "actionable_units": publishable,
        "auto_file_requested": auto_file,
        "attention_reasons": attention_reasons,
        "blockers": blockers,
        "payloads": decision_payloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--scan-date", default="")
    parser.add_argument("--auto-file", action="store_true")
    args = parser.parse_args()

    decision = build_decision(
        args.root,
        auto_file=args.auto_file,
        run_id=args.run_id,
        scan_date=args.scan_date,
        scan_root=args.scan_root,
    )
    output = args.root / "filing_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    summary = dict(decision)
    summary["payloads"] = [payload["unit_id"] for payload in decision["payloads"]]  # type: ignore[index]
    print(json.dumps(summary, indent=2))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"decision={decision['decision']}\n")
            handle.write(f"needs_attention={'true' if decision['needs_attention'] else 'false'}\n")
            handle.write(f"actionable_count={len(decision['actionable_units'])}\n")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
