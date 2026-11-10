#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Tests for checkout_torch_xpu_ops.py ref resolution.

Every CI flow (PR, nightly, on-demand, acceptance, bisect) places
torch-xpu-ops through this script, and the SHA it returns is what CMake is
pinned to. Runs against local fixture repos; no network.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import checkout_torch_xpu_ops as cx  # noqa: E402

GIT_ENV = ["-c", "user.email=ci@example.com", "-c", "user.name=ci"]


def git(cwd, *args):
    return subprocess.check_output(
        ["git", *GIT_ENV, "-C", str(cwd), *args], text=True
    ).strip()


def commit(cwd, msg):
    git(cwd, "commit", "-q", "--allow-empty", "-m", msg)
    return git(cwd, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def remote(tmp_path_factory):
    """Bare repo with main, a tag, a release branch and a PR-only commit."""
    root = tmp_path_factory.mktemp("remote")
    src = root / "src"
    subprocess.check_call(["git", "init", "-q", "-b", "main", str(src)])
    shas = {"v1": commit(src, "m1")}
    git(src, "tag", "v1")
    git(src, "branch", "release/1.0")
    shas["main"] = commit(src, "m2")
    git(src, "checkout", "-q", "--orphan", "pr")
    shas["pr"] = commit(src, "pr")
    git(src, "update-ref", "refs/pull/5/head", shas["pr"])
    git(src, "checkout", "-q", "main")
    git(src, "branch", "-q", "-D", "pr")
    bare = root / "origin.git"
    subprocess.check_call(["git", "clone", "-q", "--bare", str(src), str(bare)])
    git(src, "push", "-q", str(bare), "refs/pull/5/head:refs/pull/5/head")
    return bare.as_uri(), shas


@pytest.fixture
def pytorch_dir(tmp_path):
    (tmp_path / "third_party").mkdir()
    return tmp_path


@pytest.mark.parametrize(
    "ref,key",
    [
        ("main", "main"),
        ("v1", "v1"),
        ("release/1.0", "v1"),  # remote-only branch
        (None, "v1"),  # full SHA
        (None, "pr"),  # SHA only reachable from refs/pull
        ("refs/pull/5/head", "pr"),
    ],
)
def test_checkout_ref(remote, pytorch_dir, ref, key):
    url, shas = remote
    sha = cx.place(pytorch_dir, url, ref or shas[key])
    assert sha == shas[key]


def test_branch_resolves_without_git_errors(remote, pytorch_dir, capfd):
    url, shas = remote
    assert cx.place(pytorch_dir, url, "release/1.0") == shas["v1"]
    assert "fatal" not in capfd.readouterr().err


def test_missing_ref_fails_with_fork_hint(remote, pytorch_dir):
    url, _ = remote
    with pytest.raises(SystemExit) as exc:
        cx.place(pytorch_dir, url, "no-such-ref")
    assert "<fork-repo-url>@<ref>" in str(exc.value)


def test_source_wins_over_pinned(pytorch_dir, tmp_path_factory):
    """PR runs (incl. disable_build) test the merge commit, never the pin."""
    src = tmp_path_factory.mktemp("pr")
    subprocess.check_call(["git", "init", "-q", str(src)])
    expected = commit(src, "merge")
    (pytorch_dir / "third_party" / "xpu.txt").write_text("release/1.0\n")
    sha = cx.place(pytorch_dir, "unused", "pinned", source=src, prefer_wheel=True)
    assert sha == expected


def test_pinned_prefers_wheel(pytorch_dir, monkeypatch):
    (pytorch_dir / "third_party" / "xpu.txt").write_text("release/1.0\n")
    monkeypatch.setattr(cx, "wheel_pin", lambda: "a" * 40)
    assert cx.resolve_pinned(pytorch_dir, prefer_wheel=True) == "a" * 40
    assert cx.resolve_pinned(pytorch_dir, prefer_wheel=False) == "release/1.0"


def test_pinned_falls_back_to_source(pytorch_dir, monkeypatch):
    (pytorch_dir / "third_party" / "xpu.txt").write_text("release/1.0\r\n")
    monkeypatch.setattr(cx, "wheel_pin", lambda: "")
    assert cx.resolve_pinned(pytorch_dir, prefer_wheel=True) == "release/1.0"
