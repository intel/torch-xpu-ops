"""Configuration constants for issue-handler.

All tunable values live in config/agent_config.yml.
This module loads them once at import time and exposes them as module-level constants.
Environment variables override YAML defaults where noted.
"""
from pathlib import Path
import os

import yaml

# ---------------------------------------------------------------------------
# Load config YAML
# ---------------------------------------------------------------------------

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = AGENT_DIR / "config" / "agent_config.yml"

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Repos (env overrides YAML defaults)
# ---------------------------------------------------------------------------

_repos = _cfg.get("repos", {})
UPSTREAM_ISSUE_REPO = os.environ.get("UPSTREAM_ISSUE_REPO", _repos.get("xpu_ops_upstream", "intel/torch-xpu-ops"))
ISSUE_REPO = os.environ.get("ISSUE_REPO", _repos.get("xpu_ops_issue", "ZhaoqiongZ/torch-xpu-ops-exp"))
PRIVATE_REVIEW_REPO = os.environ.get("PRIVATE_REVIEW_REPO", _repos.get("pytorch_private", "chuanqi129/pytorch"))
PUBLIC_TARGET_REPO = os.environ.get("PUBLIC_TARGET_REPO", _repos.get("pytorch_public", "pytorch/pytorch"))

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

ISSUE_LABEL: str = _cfg.get("issue_label", "ai_generated")
STAGE_TO_LABEL: dict[str, str] = _cfg.get("stage_labels", {})
ALL_AGENT_LABELS = sorted({v for v in STAGE_TO_LABEL.values() if v})

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

_limits = _cfg.get("limits", {})
MAX_AGENT_ATTEMPTS: int = _limits.get("max_agent_attempts", 3)

# Terminal stages
TERMINAL_STAGES: set[str] = set(_cfg.get("terminal_stages", ["DONE", "SKIPPED", "NEEDS_HUMAN"]))

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

STAGE_TIMEOUTS: dict[str, int] = _cfg.get("stage_timeouts", {})

# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

_git = _cfg.get("git", {})
REVIEW_REMOTE: str = _git.get("review_remote", "review")
BRANCH_PREFIX: str = _git.get("branch_prefix", "agent/")
AGENT_PR_PREFIX: str = _git.get("pr_prefix", "[Agent]")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_paths = _cfg.get("paths", {})
PYTORCH_DIR = Path(os.environ.get("PYTORCH_DIR", os.path.expanduser(_paths.get("pytorch_dir", "~/pytorch"))))
LOG_DIR = AGENT_DIR / "logs"
<<<<<<< HEAD
=======
CONFIG_DIR = AGENT_DIR / "config"
>>>>>>> agent/fix-agent-v1
SKILLS_DIR = AGENT_DIR.parent / "skills"

# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", str(_cfg.get("poll_interval", 60))))

# ---------------------------------------------------------------------------
# Agent backend
# ---------------------------------------------------------------------------

_backend = _cfg.get("backend", {})
AGENT_BACKEND = os.environ.get("AGENT_BACKEND", _backend.get("type", "opencode"))
OPENCODE_CMD = os.environ.get("OPENCODE_CMD", _backend.get("opencode_cmd", "opencode"))
