"""Tests for agent_backend module."""
import subprocess
from unittest.mock import patch, MagicMock

import pytest
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.utils.agent_backend import (
    OpenCodeBackend, CopilotBackend, get_backend
)


class TestGetBackend:
    def test_default_returns_opencode(self):
        with patch("pytorch_agent.utils.config.AGENT_BACKEND", "opencode"):
            backend = get_backend()
            assert isinstance(backend, OpenCodeBackend)

    def test_copilot_returns_copilot(self):
        with patch("pytorch_agent.utils.config.AGENT_BACKEND", "copilot"):
            backend = get_backend()
            assert isinstance(backend, CopilotBackend)


class TestOpenCodeBackend:
    def test_run_calls_subprocess(self, tmp_path):
        backend = OpenCodeBackend()
        mock_result = MagicMock()
        mock_result.stdout = "agent output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("pytorch_agent.utils.agent_backend.LOG_DIR", tmp_path):
            output, log_path = backend.run("fix the bug", workdir="/tmp/pytorch")

        assert output == "agent output"
        assert log_path.exists()
        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == "opencode"
        assert "run" in cmd
        assert "--dir" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "fix the bug" in cmd

    def test_run_with_skill_inlines_content(self, tmp_path, monkeypatch):
        # Create a fake skill
        skill_dir = tmp_path / "pytorch-triage"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Triage skill\nDo triage.")
        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        backend = OpenCodeBackend()
        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("pytorch_agent.utils.agent_backend.SKILLS_DIR", tmp_path), \
             patch("pytorch_agent.utils.agent_backend.LOG_DIR", log_dir):
            backend.run("triage issue #5", skill="pytorch-triage")

        cmd = mock_run.call_args[0][0]
        prompt = cmd[-1]
        assert "Skill Instructions" in prompt
        assert "Triage skill" in prompt
        assert "triage issue #5" in prompt

    def test_run_raises_on_failure(self, tmp_path):
        backend = OpenCodeBackend()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error message"
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result), \
             patch("pytorch_agent.utils.agent_backend.LOG_DIR", tmp_path):
            with pytest.raises(RuntimeError, match="OpenCode failed"):
                backend.run("fail")

    def test_run_saves_log_with_issue_and_stage(self, tmp_path):
        backend = OpenCodeBackend()
        mock_result = MagicMock()
        mock_result.stdout = "fix output"
        mock_result.stderr = "some warnings"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result), \
             patch("pytorch_agent.utils.agent_backend.LOG_DIR", tmp_path):
            output, log_path = backend.run("fix it", issue=42, stage="IMPLEMENTING")

        assert "issue-42" in log_path.name
        assert "implementing" in log_path.name
        content = log_path.read_text()
        assert "fix output" in content
        assert "some warnings" in content


class TestCopilotBackend:
    def test_raises_not_implemented(self):
        backend = CopilotBackend()
        with pytest.raises(NotImplementedError):
            backend.run("anything")
