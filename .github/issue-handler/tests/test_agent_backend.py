"""Tests for agent_backend module."""
import json
import subprocess
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

import pytest
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from issue_handler.utils.agent_backend import (
    OpenCodeBackend, CopilotBackend, get_backend, parse_opencode_events
)


class TestGetBackend:
    def test_default_returns_opencode(self):
        with patch("issue_handler.utils.config.AGENT_BACKEND", "opencode"):
            backend = get_backend()
            assert isinstance(backend, OpenCodeBackend)

    def test_copilot_returns_copilot(self):
        with patch("issue_handler.utils.config.AGENT_BACKEND", "copilot"):
            backend = get_backend()
            assert isinstance(backend, CopilotBackend)


class TestParseOpenCodeEvents:
    def test_extracts_text(self):
        raw = '\n'.join([
            json.dumps({"type": "text", "part": {"text": "Hello "}}),
            json.dumps({"type": "text", "part": {"text": "world"}}),
        ])
        assert parse_opencode_events(raw) == "Hello world"

    def test_ignores_non_text(self):
        raw = json.dumps({"type": "tool_call", "name": "bash"})
        assert parse_opencode_events(raw) == ""

    def test_skips_invalid_json(self):
        raw = "not json\n" + json.dumps({"type": "text", "part": {"text": "ok"}})
        assert parse_opencode_events(raw) == "ok"


def _make_mock_popen(events: list[dict], returncode: int = 0):
    """Create a mock Popen that streams JSON events."""
    lines = [json.dumps(e) + "\n" for e in events]
    mock_proc = MagicMock()
    mock_proc.stdout = iter(lines)
    mock_proc.returncode = returncode
    mock_proc.pid = 12345
    mock_proc.wait = MagicMock()
    return mock_proc


class TestOpenCodeBackend:
    def test_run_returns_3_tuple(self, tmp_path):
        backend = OpenCodeBackend()
        events = [
            {"sessionID": "sess-123", "type": "start"},
            {"type": "text", "part": {"text": "agent output"}},
        ]
        mock_proc = _make_mock_popen(events)

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("issue_handler.utils.agent_backend.LOG_DIR", tmp_path):
            output, log_path, session_id = backend.run("fix the bug", workdir="/tmp/pytorch")

        assert output == "agent output"
        assert log_path.exists()
        assert session_id == "sess-123"

    def test_run_with_skill_adds_hint(self, tmp_path):
        skill_dir = tmp_path / "issue-triage"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Triage skill\nDo triage.")
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        backend = OpenCodeBackend()
        events = [{"type": "text", "part": {"text": "output"}}]
        mock_proc = _make_mock_popen(events)

        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("issue_handler.utils.agent_backend.SKILLS_DIR", tmp_path), \
             patch("issue_handler.utils.agent_backend.LOG_DIR", log_dir):
            backend.run("triage issue #5", skill="issue-triage")

        cmd = mock_popen.call_args[0][0]
        prompt = cmd[-1]
        assert "issue-triage" in prompt
        assert "triage issue #5" in prompt

    def test_run_raises_on_failure(self, tmp_path):
        backend = OpenCodeBackend()
        events = [{"type": "text", "part": {"text": "error"}}]
        mock_proc = _make_mock_popen(events, returncode=1)

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("issue_handler.utils.agent_backend.LOG_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="OpenCode failed"):
                backend.run("fail")

    def test_run_saves_log_with_issue_and_stage(self, tmp_path):
        backend = OpenCodeBackend()
        events = [{"type": "text", "part": {"text": "fix output"}}]
        mock_proc = _make_mock_popen(events)

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("issue_handler.utils.agent_backend.LOG_DIR", tmp_path):
            output, log_path, _ = backend.run("fix it", issue=42, stage="IMPLEMENTING")

        assert "issue-42" in log_path.name
        assert "implementing" in log_path.name
        content = log_path.read_text()
        assert "fix output" in content

    def test_on_session_start_callback(self, tmp_path):
        backend = OpenCodeBackend()
        events = [
            {"sessionID": "sess-456", "type": "start"},
            {"type": "text", "part": {"text": "done"}},
        ]
        mock_proc = _make_mock_popen(events)
        callback = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("issue_handler.utils.agent_backend.LOG_DIR", tmp_path):
            backend.run("test", on_session_start=callback)

        callback.assert_called_once_with("sess-456")


class TestCopilotBackend:
    def test_raises_not_implemented(self):
        backend = CopilotBackend()
        with pytest.raises(NotImplementedError):
            backend.run("anything")
