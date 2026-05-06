"""Configuration constants for pytorch-agent.

All tuneable values come from environment variables (see .env.example).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Repos
# ---------------------------------------------------------------------------
UPSTREAM_ISSUE_REPO: str   # e.g. "intel/torch-xpu-ops"
PRIVATE_REVIEW_REPO: str   # e.g. "yourfork/pytorch"
PUBLIC_TARGET_REPO: str    # e.g. "pytorch/pytorch"

# ---------------------------------------------------------------------------
# Labels — one label per stage; pipeline updates issues automatically
# ---------------------------------------------------------------------------
ISSUE_LABEL: str
STAGE_TO_LABEL: dict[str, str]
ALL_AGENT_LABELS: list[str]

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_REVIEW_ITERATIONS: int
MAX_AGENT_ATTEMPTS: int

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
PYTORCH_DIR: Path     # ~/pytorch by default
AGENT_DIR: Path       # .github/pytorch-agent/
LOG_DIR: Path
SKILLS_DIR: Path

# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------
REVIEW_REMOTE: str    # name of the remote pointing at PRIVATE_REVIEW_REPO
BRANCH_PREFIX: str
AGENT_PR_PREFIX: str

# ---------------------------------------------------------------------------
# Polling (temporary — replaced by webhooks in a future PR)
# ---------------------------------------------------------------------------
POLL_INTERVAL: int

# ---------------------------------------------------------------------------
# Per-stage agent timeouts (seconds)
# ---------------------------------------------------------------------------
STAGE_TIMEOUTS: dict[str, int]

# ---------------------------------------------------------------------------
# Agent backend
# ---------------------------------------------------------------------------
AGENT_BACKEND: str    # "opencode" | "copilot"
OPENCODE_CMD: str

raise NotImplementedError("config.py: stub — implementation added in PR 2")
