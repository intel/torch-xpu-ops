#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Focused tests for the XPU alignment run summary."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xpu_alignment_publish as publish  # noqa: E402
import alignment_triage as triage  # noqa: E402


def decision(**overrides):
    value = {
        "schema_version": 1,
        "run_id": "42",
        "scan_date": "2026-09-01",
        "mode": "schedule",
        "decision": "none",
        "would_decision": "none",
        "run_state": "complete",
        "collection_status": "complete",
        "collection_progress": [],
        "global_blockers": [],
        "unit_blockers": [],
        "unit_verdicts": {},
        "payloads": [],
    }
    value.update(overrides)
    return value


def test_clean_summary_is_compact():
    headline, lines, notify = publish.run_note(decision(), [], [])

    assert headline == "XPU alignment run complete"
    assert lines == [
        "- Scan date: `2026-09-01`",
        "- Collection: complete",
        "- Reviewed units: 0",
        "- New XPU tracker candidates: 0",
        "",
        "No new XPU tracker was filed or drafted.",
    ]
    assert notify is False


def test_warning_summary_counts_verdicts_and_explains_excluded_units():
    headline, lines, notify = publish.run_note(
        decision(
            run_state="complete-with-warnings",
            unit_verdicts={"issue-1": "needs-xpu-fix", "issue-3": "verification-gap"},
            unit_blockers=["scan-blocked-result:issue-2:blocked-env"],
        ),
        [{"unit_id": "issue-1", "title": "[xpu-alignment] one"}],
        [],
    )
    body = "\n".join(lines)

    assert headline == "XPU alignment run completed with warnings"
    assert "- Reviewed units: 2 (`needs-xpu-fix`: 1, `verification-gap`: 1)" in body
    assert "- Excluded units:" in body
    assert "  - `issue-2` — `blocked-env`" in body
    assert "  - `issue-3` — `verification-gap`" in body
    assert notify is True


def test_failed_partial_summary_keeps_progress_and_safe_blockers():
    headline, lines, notify = publish.run_note(
        decision(
            run_state="failed",
            decision="blocked",
            would_decision="blocked",
            collection_status="partial",
            collection_progress=[
                {
                    "source": "pytorch_issues",
                    "status": "partial",
                    "pages_completed": 3,
                    "items_fetched": 90,
                    "last_cursor": "cursor-9",
                    "rate_reset_at": "2026-09-01T10:00:00Z",
                    "error": {
                        "kind": "rate-limit",
                        "message": "HTTP 403\nretry `later`",
                    },
                }
            ],
            global_blockers=[
                "runner-log-digest-mismatch:issue-123",
                "runner-log-digest-mismatch:issue-123",
                "collection-invalid:HTTP 403\nretry `later`",
            ],
        ),
        [],
        [],
    )
    body = "\n".join(lines)

    assert headline == "XPU alignment run failed"
    assert "`pytorch_issues`: 3 page(s), 90 item(s)" in body
    assert "`cursor-9`" in body
    assert "`rate-limit`: `HTTP 403 retry 'later'`" in body
    assert body.count("`runner-log-digest-mismatch:issue-123`") == 1
    assert "`collection-invalid`: `HTTP 403 retry 'later'`" in body
    assert notify is True


def test_dry_run_summary_has_no_scheduled_marker():
    body = triage.render_run_note(
        "42", "[DRY RUN] XPU alignment run complete", ["done"], "", dry_run=True
    )

    assert "alignment-run-note" not in body
    assert "<sub>run `42`</sub>" in body


def test_scheduled_summary_marker_must_be_unique():
    marker = triage.RUN_NOTE_MARKER.format(run_id="42")
    comments = [{"id": 1, "body": marker}, {"id": 2, "body": marker}]

    try:
        triage.find_run_note(comments, "42")
    except SystemExit:
        pass
    else:
        raise AssertionError("duplicate scheduled summary markers must fail closed")


def test_state_titles_and_notifications_are_total():
    expected = {
        "complete": ("XPU alignment run complete", False),
        "complete-with-warnings": (
            "XPU alignment run completed with warnings",
            True,
        ),
        "partial": ("XPU alignment run completed with partial collection", True),
        "failed": ("XPU alignment run failed", True),
    }

    for state, outcome in expected.items():
        headline, _, notify = publish.run_note(decision(run_state=state), [], [])
        assert (headline, notify) == outcome

    headline, _, notify = publish.run_note(
        decision(mode="dry-run", decision="dry-run", run_state="failed"), [], []
    )
    assert headline == "[DRY RUN] XPU alignment run failed"
    assert notify is False


def test_global_blocker_details_are_bounded():
    blockers = [f"scan-invalid-unit:issue-{index}" for index in range(7)]
    _, lines, _ = publish.run_note(
        decision(
            run_state="failed",
            decision="blocked",
            would_decision="blocked",
            global_blockers=blockers,
        ),
        [],
        [],
    )
    body = "\n".join(lines)

    assert "`scan-invalid-unit:issue-0`" in body
    assert "`scan-invalid-unit:issue-4`" in body
    assert "scan-invalid-unit:issue-5" not in body
    assert "2 additional blocker(s) omitted" in body


def test_candidate_summary_is_a_snapshot_not_a_live_queue():
    payloads = [
        {"unit_id": "issue-1", "title": "[xpu-alignment] one"},
        {"unit_id": "issue-2", "title": "[xpu-alignment] two"},
    ]
    _, lines, notify = publish.run_note(
        decision(decision="triage", would_decision="triage", payloads=payloads),
        payloads,
        [],
    )
    body = "\n".join(lines)

    assert "Formal candidate drafts:" in body
    assert "waiting" not in body.lower()
    assert "awaiting" not in body.lower()
    assert notify is True
