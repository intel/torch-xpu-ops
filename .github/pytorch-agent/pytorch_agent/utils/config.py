"""Configuration constants for pytorch-agent.

All tunable values live in config/agent_config.yml.
This module loads them once at import time and exposes them as module-level constants.
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
# Repos (env-only — not in YAML, these are deployment-specific)
# ---------------------------------------------------------------------------

UPSTREAM_ISSUE_REPO = os.environ.get("UPSTREAM_ISSUE_REPO", "intel/torch-xpu-ops")
ISSUE_REPO = os.environ.get("ISSUE_REPO", "ZhaoqiongZ/torch-xpu-ops-exp")
PRIVATE_REVIEW_REPO = os.environ.get("PRIVATE_REVIEW_REPO", "")  # e.g. "yourfork/pytorch"
PUBLIC_TARGET_REPO = os.environ.get("PUBLIC_TARGET_REPO", "pytorch/pytorch")

# ---------------------------------------------------------------------------
# From YAML: labels
# ---------------------------------------------------------------------------

ISSUE_LABEL = "ai_generated"
STAGE_TO_LABEL: dict[str, str] = _cfg.get("stage_labels", {})
ALL_AGENT_LABELS = sorted({v for v in STAGE_TO_LABEL.values() if v})

# ---------------------------------------------------------------------------
# From YAML: limits
# ---------------------------------------------------------------------------

_limits = _cfg.get("limits", {})
MAX_REVIEW_ITERATIONS: int = _limits.get("max_review_iterations", 3)
MAX_CI_ITERATIONS: int = _limits.get("max_ci_iterations", 3)
MAX_AGENT_ATTEMPTS: int = _limits.get("max_agent_attempts", 3)

# ---------------------------------------------------------------------------
# From YAML: timeouts
# ---------------------------------------------------------------------------

STAGE_TIMEOUTS: dict[str, int] = _cfg.get("stage_timeouts", {})

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------

PYTORCH_DIR = Path(os.environ.get("PYTORCH_DIR", os.path.expanduser("~/pytorch")))
LOG_DIR = AGENT_DIR / "logs"
SKILLS_DIR = AGENT_DIR.parent / "skills"

# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

REVIEW_REMOTE = "review"
BRANCH_PREFIX = "agent/"
AGENT_PR_PREFIX = "[Agent]"

# ---------------------------------------------------------------------------
# Polling (temporary — removed when moving to webhooks)
# ---------------------------------------------------------------------------

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "60"))

# ---------------------------------------------------------------------------
# Agent backend
# ---------------------------------------------------------------------------

AGENT_BACKEND = os.environ.get("AGENT_BACKEND", "opencode")  # or "copilot"
OPENCODE_CMD = os.environ.get("OPENCODE_CMD", "opencode")
