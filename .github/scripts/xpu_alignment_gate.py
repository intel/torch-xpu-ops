#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Validate XPU alignment scan/review artifacts and choose a publishing path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

SCHEMA_VERSION = 1
UNIT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
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


def _evidence_file(run: Path, value: object, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}-path-invalid")
        return
    path = run / value
    try:
        path.resolve().relative_to(run.resolve())
    except ValueError:
        errors.append(f"{label}-path-outside-run")
        return
    if not path.is_file():
        errors.append(f"{label}-missing")


def _validate_scan(
    scan_root: Path, scan_date: str
) -> tuple[dict[str, object], Path | None, list[str], list[str]]:
    errors: list[str] = []
    scan_path = _one(scan_root, "scan.json", "scan", errors)
    if scan_path is None:
        return {}, None, [], errors
    scan = _read_json(scan_path, "scan", errors)
    if scan.get("schema_version") != SCHEMA_VERSION:
        errors.append("scan-invalid-version")

    try:
        day = date.fromisoformat(scan_date)
    except ValueError:
        errors.append("gate-invalid-scan-date")
    else:
        expected = {
            "start": f"{day.isoformat()}T00:00:00Z",
            "end": f"{(day + timedelta(days=1)).isoformat()}T00:00:00Z",
        }
        if scan.get("scan_window") != expected:
            errors.append("scan-window-mismatch")

    blockers = scan.get("blockers")
    if scan.get("status") != "complete":
        errors.append(f"scan-not-complete:{scan.get('status', 'missing')}")
    if not isinstance(blockers, list) or blockers:
        errors.append("scan-has-blockers")

    collection = scan.get("collection")
    if not isinstance(collection, dict) or collection.get("complete") is not True:
        errors.append("collection-incomplete")
    else:
        sources = collection.get("sources")
        if (
            not isinstance(sources, list)
            or any(not isinstance(source, str) for source in sources)
            or not EXPECTED_SOURCES.issubset(set(sources))
        ):
            errors.append("collection-source-coverage")
        collection_errors = collection.get("errors")
        if not isinstance(collection_errors, list) or collection_errors:
            errors.append("collection-has-errors")

    candidates = scan.get("candidates")
    if not isinstance(candidates, list):
        errors.append("scan-candidates-not-list")
        return scan, scan_path, [], errors

    seen: set[str] = set()
    actionable: list[str] = []
    run = scan_path.parent
    for entry in candidates:
        if not isinstance(entry, dict):
            errors.append("candidate-not-object")
            continue
        unit_id = entry.get("id")
        if not isinstance(unit_id, str) or not UNIT_ID_RE.fullmatch(unit_id):
            errors.append("candidate-invalid-id")
            continue
        if unit_id in seen:
            errors.append(f"candidate-duplicate:{unit_id}")
            continue
        seen.add(unit_id)
        if not isinstance(entry.get("url"), str) or not str(entry["url"]).startswith(
            "https://github.com/pytorch/pytorch/"
        ):
            errors.append(f"candidate-invalid-url:{unit_id}")
        if not str(entry.get("reason", "")).strip():
            errors.append(f"candidate-missing-reason:{unit_id}")

        triage = entry.get("triage")
        result = entry.get("local_result")
        if triage == "reject":
            if result is not None:
                errors.append(f"rejected-candidate-has-result:{unit_id}")
            continue
        if triage != "validate" or result not in LOCAL_RESULTS:
            errors.append(f"candidate-invalid-result:{unit_id}")
            continue
        if result in BLOCKED_RESULTS:
            errors.append(f"candidate-blocked:{unit_id}:{result}")
            continue

        _evidence_file(run, entry.get("reproducer"), f"reproducer:{unit_id}", errors)
        _evidence_file(run, entry.get("log"), f"log:{unit_id}", errors)
        if entry.get("target_path_verified") is not True:
            errors.append(f"candidate-target-unverified:{unit_id}")
        if not str(entry.get("oracle", "")).strip():
            errors.append(f"candidate-oracle-missing:{unit_id}")
        if result in ACTIONABLE_RESULTS:
            actionable.append(unit_id)
    return scan, scan_path, sorted(actionable), errors


def _validate_review(
    review_root: Path, scan_path: Path | None, actionable: list[str]
) -> tuple[list[dict[str, object]], dict[str, str], list[str]]:
    errors: list[str] = []
    review_path = _one(review_root, "review.json", "review", errors)
    if review_path is None:
        return [], {}, errors
    review = _read_json(review_path, "review", errors)
    if review.get("schema_version") != SCHEMA_VERSION:
        errors.append("review-invalid-version")
    if review.get("status") != "complete":
        errors.append(f"review-not-complete:{review.get('status', 'missing')}")
    blockers = review.get("blockers")
    if not isinstance(blockers, list) or blockers:
        errors.append("review-has-blockers")
    if scan_path is not None:
        digest = hashlib.sha256(scan_path.read_bytes()).hexdigest()
        if review.get("scan_sha256") != digest:
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
        unit_id = entry.get("id")
        verdict = entry.get("verdict")
        if not isinstance(unit_id, str) or unit_id in verdicts or unit_id not in actionable:
            errors.append(f"review-invalid-unit:{unit_id}")
            continue
        if verdict not in VERDICTS:
            errors.append(f"review-invalid-verdict:{unit_id}")
            continue
        verdicts[unit_id] = str(verdict)
        if entry.get("implementation_repository") not in IMPLEMENTATION_REPOSITORIES:
            errors.append(f"review-invalid-repository:{unit_id}")

        canonical = entry.get("canonical_tracker")
        if canonical is not None and (
            not isinstance(canonical, str)
            or not canonical.startswith("https://github.com/intel/torch-xpu-ops/issues/")
        ):
            errors.append(f"review-invalid-tracker:{unit_id}")
        payload = entry.get("payload")
        expects_payload = verdict == "needs-xpu-fix" and canonical is None
        if not expects_payload:
            if payload is not None:
                errors.append(f"review-unexpected-payload:{unit_id}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"review-missing-payload:{unit_id}")
            continue
        title = payload.get("title")
        body = payload.get("body")
        if not isinstance(title, str) or not title.startswith(f"{ISSUE_TITLE_PREFIX} "):
            errors.append(f"payload-invalid-title:{unit_id}")
        if not isinstance(body, str) or not body.strip():
            errors.append(f"payload-empty-body:{unit_id}")
        if payload.get("labels") != ISSUE_LABELS:
            errors.append(f"payload-invalid-labels:{unit_id}")
        payloads.append(
            {
                "unit_id": unit_id,
                "title": title,
                "body": body,
                "labels": ISSUE_LABELS,
            }
        )

    if sorted(verdicts) != actionable:
        errors.append("review-coverage-mismatch")
    return payloads, dict(sorted(verdicts.items())), errors


def build_decision(
    scan_root: Path,
    review_root: Path,
    *,
    auto_file: bool,
    producers_clean: bool,
    run_id: str,
    scan_date: str,
) -> dict[str, object]:
    _, scan_path, actionable, scan_errors = _validate_scan(scan_root, scan_date)
    payloads, verdicts, review_errors = _validate_review(
        review_root, scan_path, actionable
    )
    blockers = scan_errors + review_errors
    attention_reasons: list[str] = []
    if not producers_clean:
        attention_reasons.append("producer-job-failed")
    if "verification-gap" in verdicts.values():
        attention_reasons.append("review-verification-gap")

    if blockers:
        decision = "blocked"
        published_payloads: list[dict[str, object]] = []
    elif not payloads:
        decision = "none"
        published_payloads = []
    elif auto_file and producers_clean and len(payloads) == 1:
        decision = "file-one"
        published_payloads = payloads
    else:
        decision = "triage"
        published_payloads = payloads

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scan_date": scan_date,
        "decision": decision,
        "needs_attention": bool(blockers or attention_reasons),
        "attention_reasons": attention_reasons,
        "blockers": blockers,
        "pending_units": [],
        "mandatory_units": actionable,
        "unit_verdicts": verdicts,
        "actionable_units": sorted(str(item["unit_id"]) for item in published_payloads),
        "auto_file_requested": auto_file,
        "payloads": published_payloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--scan-date", required=True)
    parser.add_argument("--auto-file", action="store_true")
    parser.add_argument("--producers-clean", action="store_true")
    args = parser.parse_args()

    decision = build_decision(
        args.scan_root,
        args.review_root,
        auto_file=args.auto_file,
        producers_clean=args.producers_clean,
        run_id=args.run_id,
        scan_date=args.scan_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    summary = {**decision, "payloads": decision["actionable_units"]}
    print(json.dumps(summary, indent=2))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with Path(github_output).open("a", encoding="utf-8") as handle:
            handle.write(f"decision={decision['decision']}\n")
            handle.write(f"needs_attention={'true' if decision['needs_attention'] else 'false'}\n")
            handle.write(f"actionable_count={len(decision['actionable_units'])}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
