#!/usr/bin/env python3

"""Focused tests for prepare-plan validation in the deterministic runner."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xpu_alignment_runner as runner  # noqa: E402


def test_unknown_verification_is_rejected_before_execution(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    collection = root / "collection" / "collection.json"
    collection.parent.mkdir()
    collection.write_text("{}\n", encoding="utf-8")
    collection_sha256 = runner.sha256(collection)

    monkeypatch.setattr(
        runner,
        "validate_collection",
        lambda *_args, **_kwargs: (
            collection,
            {"status": "complete"},
            {"unit-1": {"id": "unit-1"}},
        ),
    )

    prepare = root / "prepare.json"
    prepare.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "blockers": [],
                "scan_window": {
                    "start": "2026-09-10T00:00:00Z",
                    "end": "2026-09-11T00:00:00Z",
                },
                "collection_sha256": collection_sha256,
                "collection_status": "complete",
                "decisions": [
                    {"id": "unit-1", "triage": "validate", "reason": "test"}
                ],
                "executions": [{"id": "unit-1", "verification": "bogus"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(runner.PlanError, match="invalid verification"):
        runner.load_prepare(root, prepare)
