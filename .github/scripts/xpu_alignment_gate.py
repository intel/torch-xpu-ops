#!/usr/bin/env python3
"""Decide what one alignment run may publish, from the scan ledger and reviewer verdicts.

Completeness is computed from ``artifacts/candidate_ledger.jsonl`` rather than
taken from a self-reported status field, so a truncated scan or a truncated
review cannot present itself as a finished day. The ledger is read from the
scan's own upload, not from the copy the reviewer repacked, because a reviewer
that can edit the record it is auditing is not an independent reviewer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from alignment_triage import ISSUE_LABELS, ISSUE_TITLE_PREFIX, UNIT_ID_RE

# Mirrors the verdict table in the official review reference.
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
ACTIONABLE_VERDICT = "needs-xpu-fix"

# A row that survives filtering has either run or not; any other value would
# fall through both the pending and the mandatory set and vanish from the day.
ALLOWED_LOCAL_STATUSES = frozenset({"done", "pending"})

LEDGER_GLOB = "**/artifacts/candidate_ledger.jsonl"
MANIFEST_GLOB = "**/reports/reviewer_manifest.json"

# The official review reference writes this line when it cannot reach a verdict,
# and says blocked outputs never unlock filing.
BLOCKED_REVIEW_RE = re.compile(r"review\s+status\s*\**\s*:\s*\**\s*blocked", re.IGNORECASE)


def _text(value: object) -> str:
    return str(value or "").strip().lower()


def load_ledger(root: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Merge every restored ledger by candidate id."""
    rows: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    paths = sorted(root.glob(LEDGER_GLOB))
    if not paths:
        return rows, ["ledger-missing"]

    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            errors.append(f"ledger-unreadable:{path}:{exc}")
            continue
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"ledger-unparsable:{path}:{number}")
                continue
            if not isinstance(row, dict):
                errors.append(f"ledger-not-an-object:{path}:{number}")
                continue
            unit_id = str(row.get("id", "")).strip()
            if not UNIT_ID_RE.fullmatch(unit_id):
                errors.append(f"ledger-invalid-id:{path}:{number}")
                continue
            normalized = {
                "title_status": _text(row.get("title_status")),
                "deep_status": _text(row.get("deep_status")),
                "local_status": _text(row.get("local_status")),
            }
            if rows.setdefault(unit_id, normalized) != normalized:
                errors.append(f"ledger-conflicting-row:{unit_id}")
    return rows, errors


def load_verdicts(root: Path) -> tuple[dict[str, str], list[str], list[str]]:
    """Merge reviewer verdicts across every restored manifest."""
    verdicts: dict[str, str] = {}
    errors: list[str] = []
    paths = sorted(root.glob(MANIFEST_GLOB))
    if not paths:
        return verdicts, ["reviewer-manifest-missing"], []

    for path in paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"manifest-unreadable:{path}:{exc}")
            continue
        if not isinstance(manifest, dict):
            errors.append(f"manifest-not-an-object:{path}")
            continue
        # The official skill owns the human-auditable Markdown; a JSON-only
        # reviewer must not be able to unlock filing without it.
        conclusions = path.parent / "review_conclusions.md"
        try:
            verdict_prose = conclusions.read_text(encoding="utf-8")
        except OSError:
            errors.append(f"manifest-missing-conclusions:{path}")
            continue
        if BLOCKED_REVIEW_RE.search(verdict_prose):
            errors.append(f"review-blocked:{conclusions}")
            continue
        units = manifest.get("units")
        if not isinstance(units, list) or not units:
            errors.append(f"manifest-missing-units:{path}")
            continue
        for entry in units:
            if not isinstance(entry, dict):
                errors.append(f"manifest-malformed-unit:{path}")
                continue
            unit_id = str(entry.get("id", "")).strip()
            verdict = _text(entry.get("verdict"))
            if not UNIT_ID_RE.fullmatch(unit_id) or verdict not in ALLOWED_VERDICTS:
                errors.append(f"manifest-invalid-unit:{unit_id or '<unnamed>'}")
                continue
            if verdicts.setdefault(unit_id, verdict) != verdict:
                errors.append(f"conflicting-verdict:{unit_id}")
    return verdicts, errors, [str(path) for path in paths]


def load_payload(root: Path, unit_id: str) -> tuple[dict[str, object] | None, str | None]:
    """Read the issue payload the scan/review phase pre-generated for one unit."""
    paths = sorted(root.glob(f"**/reports/final_issue_{unit_id}.json"))
    if len(paths) != 1:
        return None, f"payload-not-unique:{unit_id}:{len(paths)}"
    try:
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"payload-unreadable:{unit_id}:{exc}"
    if not isinstance(payload, dict):
        return None, f"payload-not-an-object:{unit_id}"
    if str(payload.get("unit_id", "")).strip() != unit_id:
        return None, f"payload-unit-mismatch:{unit_id}"
    title = str(payload.get("title", ""))
    if not title.startswith(ISSUE_TITLE_PREFIX):
        return None, f"payload-invalid-title:{unit_id}"
    if not str(payload.get("body", "")).strip():
        return None, f"payload-empty-body:{unit_id}"
    if payload.get("labels") != ISSUE_LABELS:
        return None, f"payload-invalid-labels:{unit_id}"
    return {
        "unit_id": unit_id,
        "title": title,
        "body": str(payload["body"]),
        "labels": ISSUE_LABELS,
    }, None


def build_decision(
    root: Path,
    *,
    auto_file: bool,
    run_id: str = "",
    scan_date: str = "",
    ledger_root: Path | None = None,
) -> dict[str, object]:
    ledger, ledger_errors = load_ledger(ledger_root or root)
    verdicts, review_errors, manifest_paths = load_verdicts(root)

    def survives_filtering(row: dict[str, str]) -> bool:
        return row["title_status"] == "pass" and row["deep_status"] != "reject"

    pending = sorted(
        unit
        for unit, row in ledger.items()
        if survives_filtering(row) and row["local_status"] == "pending"
    )
    mandatory = sorted(
        unit
        for unit, row in ledger.items()
        if survives_filtering(row) and row["local_status"] == "done"
    )
    invalid_statuses = sorted(
        unit
        for unit, row in ledger.items()
        if survives_filtering(row) and row["local_status"] not in ALLOWED_LOCAL_STATUSES
    )

    # A verdict the ledger never mentions means the reviewer invented a unit;
    # a mandatory row with no verdict means the review stopped early.
    coverage_gaps = [unit for unit in mandatory if unit not in verdicts]
    unknown_units = sorted(unit for unit in verdicts if unit not in ledger)

    actionable = sorted(
        unit for unit, verdict in verdicts.items() if verdict == ACTIONABLE_VERDICT
    )

    payloads: list[dict[str, object]] = []
    payload_errors: list[str] = []
    for unit in actionable:
        payload, error = load_payload(root, unit)
        if error:
            payload_errors.append(error)
        else:
            payloads.append(payload)  # type: ignore[arg-type]

    blockers = (
        ledger_errors
        + review_errors
        + [f"ledger-invalid-status:{unit}" for unit in invalid_statuses]
        + [f"coverage-gap:{unit}" for unit in coverage_gaps]
        + [f"unknown-unit:{unit}" for unit in unknown_units]
    )
    # One unusable payload must not discard the units that are fine, so it only
    # blocks the day when nothing is left to publish. It always closes the
    # unattended path, which assumes every reviewed unit reached the gate.
    if actionable and not payloads:
        blockers = blockers + payload_errors

    complete = not pending
    if blockers:
        decision = "blocked"
    elif not actionable:
        decision = "none"
    elif complete and auto_file and not payload_errors and len(payloads) == 1:
        decision = "file-one"
    else:
        decision = "triage"

    return {
        "run_id": run_id,
        "scan_date": scan_date,
        "decision": decision,
        # `blocked`, an unfinished scan and a unit that cannot be published are
        # the only conditions that should turn the run red; a busy day is normal.
        "needs_attention": bool(blockers) or not complete or bool(payload_errors),
        "scan_complete": complete,
        "pending_units": pending,
        "mandatory_units": mandatory,
        "unpublishable_units": payload_errors,
        "unit_verdicts": dict(sorted(verdicts.items())),
        "actionable_units": actionable,
        "auto_file_requested": auto_file,
        "blockers": blockers,
        "reviewer_manifests": manifest_paths,
        "payloads": payloads if decision in {"file-one", "triage"} else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    # Defaults to --root only so the tests can point both at one tree.
    parser.add_argument("--ledger-root", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--scan-date", default="")
    parser.add_argument("--auto-file", action="store_true")
    args = parser.parse_args()

    decision = build_decision(
        args.root,
        auto_file=args.auto_file,
        run_id=args.run_id,
        scan_date=args.scan_date,
        ledger_root=args.ledger_root,
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
            handle.write(
                f"needs_attention={'true' if decision['needs_attention'] else 'false'}\n"
            )
            handle.write(f"actionable_count={len(decision['actionable_units'])}\n")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
