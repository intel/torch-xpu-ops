#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Tests for apply_labels.py's target pinning.

labels.md is produced by an agent that read attacker-authored issue text, so the
issue it names is untrusted input. --expect is the control that keeps a run from
writing anywhere except the issue the command was invoked on.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "apply_labels.py"

HEADER = "label-issue: {target}"
BODY = """

| axis | value | reason |
|---|---|---|
| type | Bug | because |
"""


def write_labels_md(tmp_path, *targets):
    """Write a labels.md naming one header per target, in order."""
    text = "\n".join(HEADER.format(target=t) for t in targets) + BODY
    path = tmp_path / "labels.md"
    path.write_text(text)
    return path


def run(path, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path), *args],
        capture_output=True,
        text=True,
    )


def test_matching_target_is_accepted(tmp_path):
    path = write_labels_md(tmp_path, "intel/torch-xpu-ops#4752")
    result = run(path, "--expect", "intel/torch-xpu-ops#4752")
    assert "refusing to write" not in result.stderr


def test_mismatched_target_is_refused(tmp_path):
    path = write_labels_md(tmp_path, "intel/torch-xpu-ops#1")
    result = run(path, "--expect", "intel/torch-xpu-ops#4752")
    assert result.returncode == 2
    assert "refusing to write" in result.stderr


def test_injected_header_above_the_real_one_is_refused(tmp_path):
    # The parse binds to the first header, so a check for "a matching header
    # exists somewhere" would pass this file while the write went to #1.
    path = write_labels_md(
        tmp_path, "intel/torch-xpu-ops#1", "intel/torch-xpu-ops#4752"
    )
    result = run(path, "--expect", "intel/torch-xpu-ops#4752")
    assert result.returncode == 2
    assert "targets intel/torch-xpu-ops#1" in result.stderr


def test_other_repo_is_refused(tmp_path):
    path = write_labels_md(tmp_path, "attacker/repo#4752")
    result = run(path, "--expect", "intel/torch-xpu-ops#4752")
    assert result.returncode == 2


@pytest.mark.parametrize("target", ["intel/torch-xpu-ops#47520", "intel/torch-xpu-ops#475"])
def test_neighbouring_issue_numbers_are_refused(tmp_path, target):
    # A substring comparison would let #4752 match #47520.
    path = write_labels_md(tmp_path, target)
    result = run(path, "--expect", "intel/torch-xpu-ops#4752")
    assert result.returncode == 2


def test_without_expect_no_pinning_happens(tmp_path):
    # --expect is opt-in; callers that do not know the target (a human running
    # the script by hand) keep the previous behaviour.
    path = write_labels_md(tmp_path, "intel/torch-xpu-ops#1")
    result = run(path)
    assert "refusing to write" not in result.stderr
