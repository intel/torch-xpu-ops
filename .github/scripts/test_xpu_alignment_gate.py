#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Focused tests for alignment review ownership validation."""

import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xpu_alignment_gate as gate  # noqa: E402


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_repository(tmp_path, verdict, repository):
    collection = tmp_path / "collection.json"
    scan = tmp_path / "scan.json"
    collection.write_text("{}", encoding="utf-8")
    scan.write_text("{}", encoding="utf-8")
    payload = None
    if verdict == "needs-xpu-fix":
        payload = {
            "title": "[xpu-alignment] test",
            "body": "evidence",
            "labels": ["ai_generated"],
        }
    review = {
        "schema_version": 1,
        "collection_sha256": _sha256(collection),
        "collection_status": "complete",
        "scan_sha256": _sha256(scan),
        "status": "complete",
        "units": [{
            "id": "unit-1",
            "verdict": verdict,
            "implementation_repository": repository,
            "canonical_tracker": None,
            "payload": payload,
        }],
        "blockers": [],
    }
    (tmp_path / "review.json").write_text(json.dumps(review), encoding="utf-8")

    _, _, errors = gate._validate_review(
        tmp_path, collection, {"status": "complete"}, scan, ["unit-1"]
    )
    return errors


@pytest.mark.parametrize(
    ("verdict", "repository"),
    [
        ("needs-xpu-fix", None),
        ("needs-xpu-fix", "pytorch/pytorch"),
        ("track-upstream", None),
        ("track-upstream", "not-a-repository"),
        ("fixed", "intel/torch-xpu-ops"),
        ("duplicate", "pytorch/pytorch"),
        ("non-issue", "pytorch/pytorch"),
        ("verification-gap", "intel/torch-xpu-ops"),
    ],
)
def test_verdict_rejects_invalid_repository_shape(tmp_path, verdict, repository):
    errors = _validate_repository(tmp_path, verdict, repository)

    assert errors == ["review-invalid-repository:unit-1"]


@pytest.mark.parametrize(
    ("verdict", "repository"),
    [
        ("needs-xpu-fix", "intel/torch-xpu-ops"),
        ("track-upstream", "intel/torch-xpu-ops"),
        ("track-upstream", "pytorch/pytorch"),
        ("track-upstream", "oneapi-src/oneDNN"),
        ("fixed", None),
        ("duplicate", None),
        ("non-issue", None),
        ("verification-gap", None),
    ],
)
def test_verdict_accepts_its_repository_shape(tmp_path, verdict, repository):
    assert _validate_repository(tmp_path, verdict, repository) == []
