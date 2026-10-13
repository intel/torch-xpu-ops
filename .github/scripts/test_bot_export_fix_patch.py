#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Tests for bot_export_fix_patch.

Usage:
    python -m pytest .github/scripts/test_bot_export_fix_patch.py

Each case builds a real throwaway git repository rather than mocking git: every
bug this suite pins down lived in the git interaction itself (which repos the
walk finds, whether a branch resolves, what format-patch emits), so a mocked
git would have reproduced none of them.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bot_export_fix_patch as ex  # noqa: E402


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def make_repo(path, with_fix_branch=True, branch="agent/fix-issue-1"):
    """A repo with one base commit, optionally a fix commit on `branch`.

    Returns (base_sha, branch).
    """
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", ".")
    git(path, "config", "user.email", "t@t")
    git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("base\n")
    git(path, "add", ".")
    git(path, "commit", "-qm", "base")
    base = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if with_fix_branch:
        git(path, "checkout", "-qb", branch)
        (path / "f.txt").write_text("base\nfix\n")
        git(path, "add", ".")
        git(path, "commit", "-qm", "fix")
    return base, branch


def write_result(agent_space, repo, base, branch, verdict="PASSED",
                 slug=None, **overrides):
    """Write a fix_result JSON; `slug=None` means the single-bug file name."""
    agent_space.mkdir(parents=True, exist_ok=True)
    name = "fix_result.json" if slug is None else f"fix_result-{slug}.json"
    d = {
        "verdict": verdict,
        "target_repo": "torch-xpu-ops",
        "fix_repo_dir": str(repo),
        "branch": branch,
        "base_sha": base,
        "changed_files": ["f.txt"],
    }
    d.update(overrides)
    (agent_space / name).write_text(json.dumps(d))
    return agent_space / name


@pytest.fixture
def space(tmp_path):
    """(agent_space, out) under a shared root that doubles as the workspace."""
    return tmp_path / "agent_space_xpu", tmp_path / "out"


def patches(out):
    found = []
    for root, _, files in os.walk(out):
        found += [os.path.relpath(os.path.join(root, f), out)
                  for f in files if f.endswith(".patch")]
    return sorted(found)


def test_passed_exports_a_patch(tmp_path, space):
    agent_space, out = space
    base, branch = make_repo(tmp_path / "repo")
    write_result(agent_space, tmp_path / "repo", base, branch)

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert (made, salvaged, errors) == (1, 0, [])
    assert len(patches(out)) == 1
    assert patches(out)[0].startswith("fix-issue-1-torch-xpu-ops" + os.sep)


def test_passed_missing_base_sha_is_an_error(tmp_path, space):
    agent_space, out = space
    base, branch = make_repo(tmp_path / "repo")
    write_result(agent_space, tmp_path / "repo", base, branch, base_sha="")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert made == 0
    assert len(errors) == 1 and "missing base_sha" in errors[0]
    assert patches(out) == []


def test_passed_with_absent_branch_is_an_error(tmp_path, space):
    agent_space, out = space
    base, _ = make_repo(tmp_path / "repo")
    write_result(agent_space, tmp_path / "repo", base, "agent/fix-issue-absent")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert made == 0
    assert len(errors) == 1 and "branch not found" in errors[0]


def test_branch_without_any_fix_result_is_detected(tmp_path):
    """The lost-fix trap: a committed fix nobody recorded must not go unseen."""
    make_repo(tmp_path / "repo")

    assert any("agent/fix-issue-1" in b for b in ex.fix_branches(str(tmp_path)))


def test_no_branch_and_no_fix_result_is_clean(tmp_path):
    make_repo(tmp_path / "repo", with_fix_branch=False)

    assert ex.fix_branches(str(tmp_path)) == []


def test_branch_in_a_dot_git_file_submodule_is_detected(tmp_path):
    """A `git submodule update` checkout has a .git FILE, not a directory.

    third_party/torch-xpu-ops -- where a torch-xpu-ops fix lands in the build
    tree -- is exactly such a submodule, so matching only directories missed
    the common case.
    """
    sub = tmp_path / "outer" / "sub"
    sub.mkdir(parents=True)
    real_git = tmp_path / "realgit"
    subprocess.run(["git", "init", "-q", f"--separate-git-dir={real_git}", str(sub)],
                   check=True, capture_output=True)
    git(sub, "config", "user.email", "t@t")
    git(sub, "config", "user.name", "t")
    (sub / "g.txt").write_text("base\n")
    git(sub, "add", ".")
    git(sub, "commit", "-qm", "base")
    git(sub, "checkout", "-qb", "agent/fix-issue-9999")
    (sub / "g.txt").write_text("base\nfix\n")
    git(sub, "add", ".")
    git(sub, "commit", "-qm", "fix")

    assert (sub / ".git").is_file(), "precondition: .git must be a file here"
    assert any("agent/fix-issue-9999" in b for b in ex.fix_branches(str(tmp_path)))


def test_branch_outside_agent_space_is_detected(tmp_path):
    """fix_repo_dir is the main checkout at the workspace root.

    AGENT_SPACE is a subdirectory of it and holds only the build tree, so a
    walk rooted at AGENT_SPACE could not see the fix at all.
    """
    make_repo(tmp_path)                                  # fix at the root
    (tmp_path / "agent_space_xpu").mkdir()               # scratch dir, no repo

    assert ex.fix_branches(str(tmp_path / "agent_space_xpu")) == []
    assert any("agent/fix-issue-1" in b for b in ex.fix_branches(str(tmp_path)))


def test_pending_verify_is_salvaged_as_unverified(tmp_path, space):
    """A committed-but-unverified fix must survive, not vanish with the runner."""
    agent_space, out = space
    base, branch = make_repo(tmp_path / "repo")
    write_result(agent_space, tmp_path / "repo", base, branch,
                 verdict="PENDING_VERIFY")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert (made, salvaged, errors) == (0, 1, [])
    assert len(patches(out)) == 1
    assert patches(out)[0].startswith(os.path.join("unverified", "fix-issue-1"))


def test_non_passed_with_unusable_contract_is_not_fatal(tmp_path, space):
    agent_space, out = space
    base, _ = make_repo(tmp_path / "repo")
    write_result(agent_space, tmp_path / "repo", base, "agent/fix-issue-absent",
                 verdict="CANNOT_VERIFY")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert (made, salvaged, errors) == (0, 0, [])
    assert patches(out) == []


def test_upstream_filed_bug_is_named_after_the_upstream_issue(tmp_path, space):
    """A CI DISABLED test is filed upstream, so the patch carries THAT number.

    The bot command runs on the torch-xpu-ops tracking issue, which is only a
    queue: naming the patch after it tells a reviewer nothing about which issue
    a PR would close.
    """
    agent_space, out = space
    base, branch = make_repo(tmp_path / "repo",
                             branch="agent/fix-pytorch-issue-197334-test_foo")
    write_result(agent_space, tmp_path / "repo", base, branch, slug="test_foo")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert (made, salvaged, errors) == (1, 0, [])
    assert patches(out)[0].startswith(
        "fix-pytorch-issue-197334-test_foo-torch-xpu-ops" + os.sep)
    # the lost-fix trap must still see the non-`fix-issue-` name
    assert any(branch in b for b in ex.fix_branches(str(tmp_path)))


def test_batch_exports_one_series_per_slug(tmp_path, space):
    agent_space, out = space
    base_a, branch_a = make_repo(tmp_path / "a", branch="agent/fix-issue-1-1-aa")
    base_b, branch_b = make_repo(tmp_path / "b", branch="agent/fix-issue-1-2-bb")
    write_result(agent_space, tmp_path / "a", base_a, branch_a, slug="aa")
    write_result(agent_space, tmp_path / "b", base_b, branch_b, slug="bb")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert (made, salvaged, errors) == (2, 0, [])
    assert [p.split(os.sep)[0] for p in patches(out)] == [
        "fix-issue-1-1-aa-torch-xpu-ops", "fix-issue-1-2-bb-torch-xpu-ops"]


def test_two_repo_fix_keeps_both_halves(tmp_path, space):
    """One bug, one branch name, two repos -- the halves must not collide.

    `issue-handler` names the branch after the issue in each repo, so a fix that
    spans pytorch and torch-xpu-ops writes two units with the same branch. Both
    series restart at 0001, so a shared directory loses one half whenever the two
    commit subjects match (the 197521 grid_sample batch came one subject away).
    """
    agent_space, out = space
    branch = "agent/fix-pytorch-issue-197521"
    base_x, _ = make_repo(tmp_path / "xpuops", branch=branch)
    base_p, _ = make_repo(tmp_path / "pytorch", branch=branch)
    write_result(agent_space, tmp_path / "xpuops", base_x, branch, slug="gs")
    write_result(agent_space, tmp_path / "pytorch", base_p, branch, slug="gs-pytorch",
                 target_repo="pytorch")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert (made, salvaged, errors) == (2, 0, [])
    assert [p.split(os.sep)[0] for p in patches(out)] == [
        "fix-pytorch-issue-197521-pytorch", "fix-pytorch-issue-197521-torch-xpu-ops"]


def test_two_units_claiming_one_directory_is_an_error(tmp_path, space):
    """The guard behind the suffix: a collision must never be a silent green.

    Both halves here carry the same branch AND the same target_repo, which means
    the agent broke its own branch naming. Exporting anyway would write both
    series into one directory, and `format-patch` restarting at 0001 would drop
    one of them while this function counted two.
    """
    agent_space, out = space
    branch = "agent/fix-issue-1"
    base_a, _ = make_repo(tmp_path / "a", branch=branch)
    base_b, _ = make_repo(tmp_path / "b", branch=branch)
    write_result(agent_space, tmp_path / "a", base_a, branch, slug="aa")
    write_result(agent_space, tmp_path / "b", base_b, branch, slug="bb")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert made == 1
    assert len(errors) == 1 and "already another unit's" in errors[0]


def test_unparseable_fix_result_is_an_error(tmp_path, space):
    agent_space, out = space
    agent_space.mkdir(parents=True)
    (agent_space / "fix_result.json").write_text("{not json")

    made, salvaged, errors = ex.export(str(agent_space), str(out), str(tmp_path))

    assert made == 0
    assert len(errors) == 1 and "unparseable" in errors[0]
