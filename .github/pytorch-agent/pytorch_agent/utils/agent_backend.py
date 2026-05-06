"""AI agent backend abstraction.

Backends
--------
OpenCodeBackend  —  wraps the `opencode` CLI, streams JSON events,
                    captures session ID and output text, writes a log file.

Usage
-----
    backend = get_backend()
    output, log_path, session_id = backend.run(
        prompt,
        workdir=str(PYTORCH_DIR),
        skill="xpu-ops-implement",
        timeout=3600,
        issue=tracked.source_number,
        stage="IMPLEMENTING",
        on_session_start=lambda sid: ...,
    )
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Callable


class AgentBackend:
    """Abstract base class for AI coding agent backends."""

    def run(
        self,
        prompt: str,
        *,
        workdir: str,
        skill: str | None = None,
        timeout: int = 3600,
        issue: int | None = None,
        stage: str | None = None,
        on_session_start: Callable[[str], None] | None = None,
    ) -> tuple[str, Path, str | None]:
        """Run the agent on *prompt* and return (output_text, log_path, session_id)."""
        raise NotImplementedError


class OpenCodeBackend(AgentBackend):
    """Runs `opencode run --format json …` and streams the response."""

    def run(self, prompt: str, *, workdir: str, skill: str | None = None,
            timeout: int = 3600, issue: int | None = None,
            stage: str | None = None,
            on_session_start: Callable[[str], None] | None = None,
            ) -> tuple[str, Path, str | None]:
        raise NotImplementedError


def get_backend() -> AgentBackend:
    """Return the configured backend (OpenCodeBackend by default)."""
    raise NotImplementedError


def parse_opencode_events(raw_json_lines: str) -> str:
    """Extract assistant text from a newline-delimited stream of opencode JSON events."""
    raise NotImplementedError
