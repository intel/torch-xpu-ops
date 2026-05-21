"""Configuration constants for pytorch-agent."""
from pathlib import Path
import os

# Repos (configure via env — no personal forks hardcoded)
UPSTREAM_ISSUE_REPO = os.environ.get("UPSTREAM_ISSUE_REPO", "intel/torch-xpu-ops")
PRIVATE_REVIEW_REPO = os.environ.get("PRIVATE_REVIEW_REPO", "")  # e.g. "yourfork/pytorch"
PUBLIC_TARGET_REPO = os.environ.get("PUBLIC_TARGET_REPO", "pytorch/pytorch")

# Labels
ISSUE_LABEL = "ai_generated"
STAGE_TO_LABEL = {
    "DISCOVERED": "agent:active",
    "IMPLEMENTING": "agent:active",
    "IN_REVIEW": "agent:blocked",
    "PUBLIC_PR": "agent:blocked",
    "CI_WATCH": "agent:active",
    "MERGED": "agent:active",
    
    "DONE": "agent:done",
    "SKIPPED": "agent:skipped",
    "NEEDS_HUMAN": "agent:needs-human",
}
ALL_AGENT_LABELS = sorted(set(STAGE_TO_LABEL.values()))

# Limits
MAX_REVIEW_ITERATIONS = 3
MAX_AGENT_ATTEMPTS = 3

# Local paths
PYTORCH_DIR = Path(os.environ.get("PYTORCH_DIR", os.path.expanduser("~/pytorch")))
AGENT_DIR = Path(__file__).resolve().parent.parent.parent  # .github/pytorch-agent/
LOG_DIR = AGENT_DIR / "logs"
SKILLS_DIR = AGENT_DIR.parent / "skills"

# Git
REVIEW_REMOTE = "review"
BRANCH_PREFIX = "agent/"
AGENT_PR_PREFIX = "[Agent]"

# Polling (temporary — removed when moving to webhooks)
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "60"))

# Per-stage timeouts (seconds)
STAGE_TIMEOUTS = {
    "TRIAGING": 300,
    "IMPLEMENTING": 3600,
    "IN_REVIEW": 1800,
    "CI_WATCH": 600,
}

# Agent backend
AGENT_BACKEND = os.environ.get("AGENT_BACKEND", "opencode")  # or "copilot"
OPENCODE_CMD = os.environ.get("OPENCODE_CMD", "opencode")
