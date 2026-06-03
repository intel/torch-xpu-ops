# Copyright 2024-2026 Intel Corporation
# Co-authored with GitHub Copilot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Pipeline stage / skill / target-repo definitions.

These were previously string literals scattered across orchestrator.py,
triage_agent.py, code_fix.py and verify_*.py.  A single typo (``"TRIGED"``
vs ``"TRIAGED"``) would silently jam an issue mid-pipeline forever, so
they're consolidated here as ``str, Enum`` pairs (which still compare equal to
the underlying string — every existing comparison ``stage == "DONE"`` keeps
working unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import (
    ISSUE_REPO, PRIVATE_REVIEW_REPO, PYTORCH_DIR, REVIEW_REMOTE,
    TORCH_XPU_OPS_DIR,
)


class Stage(str, Enum):
    """Pipeline stages stored in the issue body as ``<!-- agent:status:X -->``."""

    DISCOVERED = "DISCOVERED"
    UPSTREAM_VERIFYING = "UPSTREAM_VERIFYING"   # checking out + verifying an upstream PR
    WAITING_UPSTREAM = "WAITING_UPSTREAM"       # verified locally; awaiting upstream merge
    TRIAGING = "TRIAGING"        # never set today; kept for forward-compat
    TRIAGED = "TRIAGED"
    IMPLEMENTING = "IMPLEMENTING"
    IN_REVIEW = "IN_REVIEW"
    PUBLIC_PR = "PUBLIC_PR"
    CI_WATCH = "CI_WATCH"
    MERGED = "MERGED"
    DONE = "DONE"
    SKIPPED = "SKIPPED"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    AWAITING_COPILOT = "AWAITING_COPILOT"   # Copilot was assigned, waiting on PR


class Skill(str, Enum):
    """Names of the OpenCode skills the agents call into."""

    FORMAT = "issue-format"
    TRIAGE = "issue-triage"
    FIX = "issue-fix"
    TEST_VERIFICATION = "test-verification"
    UPSTREAM_PR_ANALYSIS = "upstream-pr-analysis"


class TargetRepo(str, Enum):
    """Where a fix should land."""

    PYTORCH = "pytorch"
    TORCH_XPU_OPS = "torch-xpu-ops"


# ---------------------------------------------------------------------------
# Per-target-repo profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoProfile:
    """Bundles together everything that varies between fix targets.

    Avoids the parallel-string-tangle that existed in code_fix.run() where
    workdir / remote / base_ref / pr_repo were all derived from a chain of
    ``if target_repo == "torch-xpu-ops"`` branches.
    """

    target: TargetRepo
    workdir: Path        # default working directory (may be replaced by a worktree)
    remote: str          # git remote to push the fix branch to
    base_ref: str        # branch on that remote (push target / PR base name)
    pr_repo: str         # owner/repo where the PR is opened
    diff_base: str       # full ref to diff/reset against (e.g. "upstream/main").
    # MUST match the ref the agent branch is created from
    # in code_fix._setup_*_branch -- otherwise a stale push
    # remote produces a bogus huge diff that masquerades
    # as "already fixed" and triggers a wasted build.


def repo_profile(target: str | TargetRepo) -> RepoProfile:
    """Look up the profile for a given target repo string."""
    t = TargetRepo(target) if not isinstance(target, TargetRepo) else target
    if t is TargetRepo.TORCH_XPU_OPS:
        return RepoProfile(
            target=t,
            workdir=TORCH_XPU_OPS_DIR,
            remote="origin",            # intel/torch-xpu-ops
            base_ref="main",
            pr_repo=ISSUE_REPO,         # PR opens against the same repo
            diff_base="origin/main",    # xpu-ops branches off origin/main
        )
    return RepoProfile(
        target=TargetRepo.PYTORCH,
        workdir=PYTORCH_DIR,
        remote=REVIEW_REMOTE,           # chuanqi129/pytorch (push target)
        base_ref="main",
        pr_repo=PRIVATE_REVIEW_REPO,
        diff_base="upstream/main",      # pytorch branches off upstream/main
                                        # (see code_fix._setup_pytorch_branch)
    )
