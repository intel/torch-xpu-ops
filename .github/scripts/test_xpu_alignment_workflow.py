#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Focused regression tests for the XPU alignment artifact pipeline."""

import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xpu_alignment_gate as gate  # noqa: E402
import xpu_alignment_runner as runner  # noqa: E402


UPSTREAM_COMMIT = "a" * 40
XPU_COMMIT = "b" * 40
TARGET_PATH = "src/ATen/native/xpu/sycl/ExampleKernels.cpp"


def _source(root, repository, commit, path, snapshot):
    snapshot_path = root / snapshot
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(f"source for {repository}:{path}\n", encoding="utf-8")
    return {
        "repository": repository,
        "commit": commit,
        "path": path,
        "snapshot": snapshot,
        "sha256": gate._sha256(snapshot_path),
    }


def _static_execution(root):
    return {
        "id": "static-unit",
        "verification": "static",
        "oracle": "The upstream and XPU checks must match.",
        "target_path": TARGET_PATH,
        "upstream_source": _source(
            root,
            "pytorch/pytorch",
            UPSTREAM_COMMIT,
            "aten/src/ATen/native/Example.cpp",
            "evidence/static-unit-upstream.txt",
        ),
        "xpu_source": _source(
            root,
            "intel/torch-xpu-ops",
            XPU_COMMIT,
            TARGET_PATH,
            "evidence/static-unit-xpu.txt",
        ),
    }


def _review(root, tracker, tracker_state, payload):
    value = {
        "schema_version": 1,
        "collection_status": "complete",
        "status": "complete",
        "units": [
            {
                "id": "unit",
                "verdict": "needs-xpu-fix",
                "implementation_repository": "intel/torch-xpu-ops",
                "canonical_tracker": tracker,
                "canonical_tracker_state": tracker_state,
                "payload": payload,
            }
        ],
        "blockers": [],
    }
    (root / "review.json").write_text(json.dumps(value), encoding="utf-8")
    return gate._validate_review(root, None, {"status": "complete"}, None, ["unit"])


def test_static_only_plan_skips_xpu_environment_probe(tmp_path, monkeypatch):
    prepare = tmp_path / "prepare.json"
    prepare.write_text('{"collection_sha256":"digest"}', encoding="utf-8")
    python = tmp_path / "python"
    python.touch()

    def unexpected_probe(*_args):
        raise AssertionError("static-only plans must not probe the XPU runtime")

    monkeypatch.setattr(runner, "probe_environment", unexpected_probe)
    monkeypatch.setattr(
        runner,
        "_become_child_subreaper",
        lambda: (_ for _ in ()).throw(
            AssertionError("static-only plans must not establish a process boundary")
        ),
    )

    result = runner.run_plan(tmp_path, python, prepare, [])

    assert result["environment"] is None
    assert result["results"] == []


def test_static_prepare_loads_without_scripts_directory(tmp_path, monkeypatch):
    execution = _static_execution(tmp_path)
    collection_path = tmp_path / "collection.json"
    collection_path.write_text("{}", encoding="utf-8")
    prepare = {
        "schema_version": 1,
        "status": "complete",
        "scan_window": {
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-02T00:00:00Z",
        },
        "collection_sha256": runner.sha256(collection_path),
        "collection_status": "complete",
        "decisions": [
            {"id": "static-unit", "triage": "validate", "reason": "source divergence"}
        ],
        "executions": [execution],
        "blockers": [],
    }
    prepare_path = tmp_path / "prepare.json"
    prepare_path.write_text(json.dumps(prepare), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "validate_collection",
        lambda *_args: (
            collection_path,
            {"status": "complete"},
            {"static-unit": {}},
        ),
    )

    assert runner.load_prepare(tmp_path, prepare_path) == []
    assert not (tmp_path / "scripts").exists()


def test_gate_accepts_static_only_runner_without_environment(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "environment": None,
                "results": [],
            }
        ),
        encoding="utf-8",
    )

    path, environment, results, errors = gate._validate_runner(
        tmp_path,
        None,
        None,
        {"static-unit": {"verification": "static"}},
    )

    assert path == results_path
    assert environment is None
    assert results == {}
    assert errors == []


def test_static_prepare_validates_source_snapshot_digests(tmp_path):
    execution = _static_execution(tmp_path)
    prepare = {
        "schema_version": 1,
        "status": "complete",
        "scan_window": {
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-02T00:00:00Z",
        },
        "collection_status": "complete",
        "decisions": [
            {"id": "static-unit", "triage": "validate", "reason": "source divergence"}
        ],
        "executions": [execution],
        "blockers": [],
    }
    (tmp_path / "prepare.json").write_text(json.dumps(prepare), encoding="utf-8")
    collection = {
        "repository": "pytorch/pytorch",
        "status": "complete",
        "snapshot": {"default_branch_head": UPSTREAM_COMMIT},
    }
    inventory = {"static-unit": {}}

    _, _, _, errors = gate._validate_prepare(
        tmp_path, "2026-09-01", None, collection, inventory
    )
    assert errors == []

    (tmp_path / execution["upstream_source"]["snapshot"]).write_text(
        "tampered\n", encoding="utf-8"
    )
    _, _, _, errors = gate._validate_prepare(
        tmp_path, "2026-09-01", None, collection, inventory
    )
    assert "execution-upstream-source:static-unit-digest-mismatch" in errors


def test_static_scan_must_reference_both_validated_snapshots(tmp_path):
    execution = _static_execution(tmp_path)
    scan = {
        "schema_version": 1,
        "collection_status": "complete",
        "environment": None,
        "status": "complete",
        "candidates": [
            {
                "id": "static-unit",
                "local_result": "confirmed",
                "target_path_verified": True,
                "evidence": "verified",
            }
        ],
        "blockers": [],
    }
    scan_path = tmp_path / "scan.json"
    scan_path.write_text(json.dumps(scan), encoding="utf-8")

    _, _, _, errors = gate._validate_scan(
        tmp_path,
        tmp_path,
        None,
        {"status": "complete"},
        None,
        None,
        None,
        {"static-unit": execution},
        {},
    )
    assert "scan-static-evidence-mismatch:static-unit" in errors

    scan["candidates"][0]["evidence"] = {
        "upstream_source": execution["upstream_source"]["snapshot"],
        "xpu_source": execution["xpu_source"]["snapshot"],
    }
    scan_path.write_text(json.dumps(scan), encoding="utf-8")
    _, actionable, _, errors = gate._validate_scan(
        tmp_path,
        tmp_path,
        None,
        {"status": "complete"},
        None,
        None,
        None,
        {"static-unit": execution},
        {},
    )
    assert errors == []
    assert actionable == ["static-unit"]


def test_open_external_tracker_does_not_replace_payload(tmp_path):
    payload = {
        "title": "[xpu-alignment] external tracker",
        "body": "The upstream tracker does not own the XPU implementation.",
        "labels": ["ai_generated"],
    }
    payloads, _, errors = _review(
        tmp_path,
        "https://github.com/pytorch/pytorch/issues/123",
        "open",
        payload,
    )

    assert errors == []
    assert len(payloads) == 1


def test_closed_tracker_payload_must_cite_tracker(tmp_path):
    tracker = "https://github.com/intel/torch-xpu-ops/issues/123"
    payload = {
        "title": "[xpu-alignment] closed tracker",
        "body": "This payload omits the prior tracker.",
        "labels": ["ai_generated"],
    }
    _, _, errors = _review(tmp_path, tracker, "closed", payload)
    assert "payload-missing-canonical-tracker:unit" in errors

    payload["body"] += f" Prior tracker: {tracker}"
    payloads, _, errors = _review(tmp_path, tracker, "closed", payload)
    assert errors == []
    assert len(payloads) == 1
