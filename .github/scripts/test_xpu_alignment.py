# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Exercise the alignment scope policy across preparation, gating, and publishing."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import xpu_alignment_gate as gate
import xpu_alignment_publish as publish
from alignment_triage import find_run_note, find_unit_comments, render_run_note
from xpu_alignment_collect import FetchError, collect, sha256
from xpu_alignment_runner import PlanError, load_prepare


SCAN_DATE = "2026-11-02"
WINDOW = {"start": f"{SCAN_DATE}T00:00:00Z", "end": "2026-11-03T00:00:00Z"}
COMMIT_ID = "commit-" + "b" * 40


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class InventoryGitHub:
    def __init__(self, partial):
        self.partial = partial

    def snapshot(self, repository):
        return {"default_branch": "main", "default_branch_head": "a" * 40}

    def page(self, repository, source, cursor, snapshot, scan_window):
        if self.partial and source == "prs-merged":
            raise FetchError("timeout", "upstream unavailable")
        ids = {
            "issues-created": ["issue-1", "issue-2"],
            "prs-created": ["pr-3"],
            "prs-merged": ["pr-3"],
            "default-branch-commits": [COMMIT_ID],
        }[source]
        nodes = []
        for unit_id in ids:
            kind, number = unit_id.split("-", 1)
            path = {"issue": "issues", "pr": "pull", "commit": "commit"}[kind]
            nodes.append(
                {
                    "id": unit_id,
                    "kind": kind,
                    "title": f"Upstream {unit_id}",
                    "url": f"https://github.com/pytorch/pytorch/{path}/{number}",
                    "event_at": f"{SCAN_DATE}T12:00:00Z",
                }
            )
        return {
            "nodes": nodes,
            "page_info": {"has_next_page": False, "end_cursor": None},
            "rate": {"remaining": 1000, "reset_at": None},
            "raw": {"nodes": nodes},
        }


def prepare_run(root, selected, *, partial=False):
    collection = collect("pytorch/pytorch", WINDOW, root, InventoryGitHub(partial))
    decisions, executions = [], []
    for item in collection["inventory"]:
        unit_id = item["id"]
        triage, categories = selected.get(unit_id, ("reject", []))
        decisions.append(
            {
                "id": unit_id,
                "triage": triage,
                "categories": categories,
                "reason": f"Summary for {unit_id}",
            }
        )
        if triage == "validate":
            script = root / f"scripts/{unit_id}.py"
            script.parent.mkdir(exist_ok=True)
            script.write_text("print('target reached')\n", encoding="utf-8")
            executions.append(
                {
                    "id": unit_id,
                    "script": str(script.relative_to(root)),
                    "script_sha256": sha256(script),
                    "timeout_seconds": 120,
                    "oracle": "upstream assertion",
                    "target_path": "single-device target",
                }
            )
    prepare = {
        "schema_version": 1,
        "status": "complete",
        "scan_window": WINDOW,
        "collection_sha256": sha256(root / "collection/collection.json"),
        "collection_status": collection["status"],
        "decisions": decisions,
        "executions": executions,
        "blockers": [],
    }
    write_json(root / "prepare.json", prepare)
    return prepare


def decision(root, *, clean=True, mode="schedule"):
    return gate.build_decision(
        root,
        root,
        root,
        root,
        root,
        mode=mode,
        producers_clean=clean,
        run_id="123",
        scan_date=SCAN_DATE,
    )


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("mode", ["schedule", "dry-run"])
def test_record_only_run_needs_no_xpu_or_downstream_artifacts(tmp_path, partial, mode):
    prepare_run(
        tmp_path,
        {
            "issue-1": ("record-only", ["training-performance"]),
            "pr-3": ("record-only", ["distributed"]),
            COMMIT_ID: ("record-only", ["distributed"]),
        },
        partial=partial,
    )
    assert load_prepare(tmp_path, tmp_path / "prepare.json") == []
    result = decision(tmp_path, mode=mode)
    assert result["run_state"] == ("partial" if partial else "complete")
    assert result["global_blockers"] == []
    assert result["payloads"] == []
    assert len(result["recorded_units"]) == 3
    assert len({item["id"] for item in result["recorded_units"]}) == 3
    headline, lines, _ = publish.run_note(result, [], [], [])
    assert ("[DRY RUN]" in headline) == (mode == "dry-run")
    text = "\n".join(lines)
    for item in result["recorded_units"]:
        assert item["url"] in text
        assert item["summary"] in text
    assert "XPU not validated" in text


def test_mixed_record_survives_xpu_job_failure(tmp_path):
    prepare_run(
        tmp_path,
        {
            "issue-1": ("validate", ["training-functional", "training-performance"]),
            "issue-2": ("record-only", ["distributed", "training-functional"]),
        },
    )
    entries = load_prepare(tmp_path, tmp_path / "prepare.json")
    assert [item["id"] for item in entries] == ["issue-1"]
    result = decision(tmp_path, clean=False)
    assert result["run_state"] == "failed"
    assert result["payloads"] == []
    assert "producer-job-failed" in result["global_blockers"]
    assert {item["id"] for item in result["recorded_units"]} == {"issue-1", "issue-2"}
    mixed = next(item for item in result["recorded_units"] if item["id"] == "issue-1")
    assert mixed["categories"] == ["training-performance"]
    headline, lines, _ = publish.run_note(result, [], [], [])
    assert "failed" in headline
    assert "Recorded observations (XPU not validated):" in lines


@pytest.mark.parametrize(
    "category",
    ["training-functional", "inference-functional", "inference-performance"],
)
def test_validation_categories_still_require_runner_evidence(tmp_path, category):
    prepare_run(tmp_path, {"issue-1": ("validate", [category])})
    assert len(load_prepare(tmp_path, tmp_path / "prepare.json")) == 1
    result = decision(tmp_path)
    assert result["run_state"] == "failed"
    assert result["recorded_units"] == []
    assert result["payloads"] == []
    assert "runner-count:0" in result["global_blockers"]


@pytest.mark.parametrize(
    ("triage", "categories"),
    [
        ("validate", ["training-performance"]),
        ("validate", ["distributed", "training-functional"]),
        ("record-only", ["inference-performance"]),
        ("validate", []),
        ("record-only", ["unknown"]),
    ],
)
def test_invalid_scope_blocks_execution_and_publication(tmp_path, triage, categories):
    prepare_run(tmp_path, {"issue-1": (triage, categories)})
    with pytest.raises(PlanError):
        load_prepare(tmp_path, tmp_path / "prepare.json")
    result = decision(tmp_path)
    assert result["run_state"] == "failed"
    assert result["recorded_units"] == []
    assert result["payloads"] == []


@pytest.mark.parametrize("validate", [False, True])
def test_preflight_selects_jobs_without_runtime(tmp_path, validate):
    choice = (
        ("validate", ["training-functional"])
        if validate
        else ("record-only", ["distributed"])
    )
    prepare_run(tmp_path, {"issue-1": choice})
    output = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("xpu_alignment_runner.py")),
            "--root",
            str(tmp_path),
            "--prepare",
            "prepare.json",
            "--check-only",
        ],
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text() == f"needs_validation={str(validate).lower()}\n"
    assert not (tmp_path / "runner").exists()


def test_bad_collection_cannot_publish_records(tmp_path):
    prepare_run(tmp_path, {"issue-1": ("record-only", ["distributed"])})
    page = next((tmp_path / "collection/pages").rglob("*.json"))
    page.write_text("{}", encoding="utf-8")
    result = decision(tmp_path)
    assert result["run_state"] == "failed"
    assert result["recorded_units"] == []


@pytest.mark.parametrize("local_result", ["confirmed", "needs-performance-harness"])
def test_validated_mixed_unit_keeps_records_separate_from_findings(tmp_path, local_result):
    blocked = local_result == "needs-performance-harness"
    category = "inference-performance" if blocked else "training-functional"
    prepare = prepare_run(
        tmp_path,
        {
            "issue-1": ("validate", [category, "training-performance"]),
            "issue-2": ("record-only", ["distributed"]),
        },
    )
    log = tmp_path / "runner/logs/issue-1.log"
    log.parent.mkdir(parents=True)
    log.write_text("Target reached; upstream oracle observed\n", encoding="utf-8")
    environment = {
        "python_executable": "/usr/bin/python3",
        "python_version": "3.13",
        "torch_version": "nightly+xpu",
        "torch_path": "/runtime/torch/__init__.py",
        "xpu_available": True,
        "xpu_device_name": "test XPU",
        "environment_warnings": [],
    }
    runner = {
        "schema_version": 1,
        "status": "complete",
        "collection_sha256": prepare["collection_sha256"],
        "prepare_sha256": sha256(tmp_path / "prepare.json"),
        "environment": environment,
        "results": [
            {
                "id": "issue-1",
                "script_sha256": prepare["executions"][0]["script_sha256"],
                "log": "runner/logs/issue-1.log",
                "log_sha256": sha256(log),
                "command": ["python3", "scripts/issue-1.py"],
                "timed_out": False,
                "returncode": 0,
                "error": None,
            }
        ],
    }
    write_json(tmp_path / "runner/results.json", runner)
    scan = {
        "schema_version": 1,
        "status": "incomplete" if blocked else "complete",
        "collection_sha256": prepare["collection_sha256"],
        "collection_status": "complete",
        "prepare_sha256": sha256(tmp_path / "prepare.json"),
        "runner_sha256": sha256(tmp_path / "runner/results.json"),
        "environment": environment,
        "candidates": [
            {
                "id": "issue-1",
                "local_result": local_result,
                "target_path_verified": True,
                "evidence": "runner/logs/issue-1.log",
            }
        ],
        "blockers": (
            [
                {
                    "id": "issue-1",
                    "local_result": local_result,
                    "detail": "Missing comparable baseline",
                    "evidence": "runner/logs/issue-1.log",
                }
            ]
            if blocked
            else []
        ),
    }
    write_json(tmp_path / "scan.json", scan)
    review = {
        "schema_version": 1,
        "status": "complete",
        "collection_sha256": prepare["collection_sha256"],
        "collection_status": "complete",
        "scan_sha256": sha256(tmp_path / "scan.json"),
        "blockers": [],
        "units": (
            []
            if blocked
            else [
                {
                    "id": "issue-1",
                    "verdict": "needs-xpu-fix",
                    "implementation_repository": "intel/torch-xpu-ops",
                    "payload": {
                        "title": "[xpu-alignment] Functional defect",
                        "body": "Runner-backed functional finding",
                        "labels": ["ai_generated"],
                    },
                }
            ]
        ),
    }
    write_json(tmp_path / "review/review.json", review)
    result = decision(tmp_path)
    assert result["global_blockers"] == []
    assert len(result["recorded_units"]) == 2
    assert result["decision"] == ("none" if blocked else "auto-file")
    assert result["run_state"] == ("complete-with-warnings" if blocked else "complete")
    assert len(result["payloads"]) == (0 if blocked else 1)


@pytest.mark.parametrize("failed", [False, True])
def test_publisher_posts_records_when_validation_fails(tmp_path, monkeypatch, failed):
    selected = {"issue-1": ("record-only", ["training-performance"])}
    if failed:
        selected["issue-2"] = ("validate", ["training-functional"])
    prepare_run(tmp_path, selected)
    result = decision(tmp_path, clean=not failed)
    decision_path = tmp_path / "decision.json"
    write_json(decision_path, result)
    comments = []
    monkeypatch.setattr(publish, "list_comments", lambda *args: [])
    monkeypatch.setattr(
        publish, "post_comment", lambda repo, issue, body: comments.append(body)
    )
    monkeypatch.setattr(
        publish, "create_issue", lambda *args: pytest.fail("unexpected tracker")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish",
            "--repo",
            "intel/torch-xpu-ops",
            "--triage-issue",
            "5018",
            "--run-url",
            "https://github.com/intel/torch-xpu-ops/actions/runs/123",
            "--decision",
            str(decision_path),
        ],
    )
    assert publish.main() == 0
    assert len(comments) == 1
    assert "Summary for issue-1" in comments[0]
    assert "XPU not validated" in comments[0]
    assert ("XPU alignment run failed" in comments[0]) == failed


@pytest.mark.parametrize("field", ["title", "summary"])
def test_recorded_text_cannot_impersonate_comment_markers(tmp_path, field):
    prepare_run(tmp_path, {"issue-1": ("record-only", ["distributed"])})
    result = decision(tmp_path)
    injected = "source <!-- alignment-unit: issue-99 --> <!-- alignment-run-note: 999 -->"
    result["recorded_units"][0][field] = injected
    headline, lines, _ = publish.run_note(result, [], [], [])
    body = render_run_note(
        "123",
        "https://github.com/intel/torch-xpu-ops/actions/runs/123",
        headline,
        lines,
        "",
    )
    comment = {"id": 321, "body": body, "user": {"login": "torchxpubot"}}
    assert find_unit_comments([comment], "issue-99") == []
    assert find_run_note([comment], "999") is None
    assert find_run_note([comment], "123") == comment


def test_long_record_list_has_bounded_display_and_retains_artifact_data(tmp_path):
    prepare_run(tmp_path, {"issue-1": ("record-only", ["distributed"])})
    result = decision(tmp_path)
    records = [
        {
            "id": f"issue-{index}",
            "title": "Upstream " + "算" * 190,
            "url": f"https://github.com/pytorch/pytorch/issues/{index}",
            "categories": ["distributed"],
            "summary": "Summary " + "例" * 490,
        }
        for index in range(200)
    ]
    result["recorded_units"] = records
    original = json.dumps(result)
    headline, lines, _ = publish.run_note(result, [], [], [])
    body = render_run_note(
        "123",
        "https://github.com/intel/torch-xpu-ops/actions/runs/123",
        headline,
        lines,
        "",
    )
    displayed = sum("XPU not validated:" in line for line in lines)
    assert 0 < displayed < len(records)
    assert len(body.encode("utf-8")) < 60_000
    assert f"{len(records) - displayed} additional observations" in body
    assert "filing_decision.json" in body
    assert json.dumps(result) == original
