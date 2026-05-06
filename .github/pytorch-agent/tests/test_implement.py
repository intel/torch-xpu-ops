"""Tests for fixing_steps/implement.py."""
import subprocess
import sys
import os
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.utils.state import TrackedIssue
from pytorch_agent.fixing_steps.implement import run


class TestImplementRun:
    def _make_tracked(self, **kwargs):
        defaults = dict(
            source_repo="intel/torch-xpu-ops",
            source_number=42,
            title="Fix XPU conv",
            stage="IMPLEMENTING",
            branch="agent/issue-42",
            attempt_count=0,
            _state_comment_id=100,
        )
        defaults.update(kwargs)
        return TrackedIssue(**defaults)

    @patch("pytorch_agent.fixing_steps.implement.get_backend")
    @patch("pytorch_agent.fixing_steps.implement.gh")
    @patch("pytorch_agent.fixing_steps.implement.subprocess.run")
    @patch("pytorch_agent.fixing_steps.implement.save_state")
    @patch("pytorch_agent.fixing_steps.implement.update_stage")
    def test_happy_path(self, mock_update, mock_save, mock_subproc, mock_gh, mock_backend):
        tracked = self._make_tracked()
        mock_gh.get_issue_detail.return_value = {"title": "Fix", "body": "details"}
        mock_gh.create_draft_pr.return_value = {"number": 5, "html_url": "https://..."}
        mock_backend_inst = MagicMock()
        mock_backend_inst.run.return_value = "fixed"
        mock_backend.return_value = mock_backend_inst
        mock_subproc.return_value = MagicMock(stdout="abc123", returncode=0)

        run(tracked)

        assert tracked.attempt_count == 1
        mock_update.assert_called_once()
        assert "IN_REVIEW" in str(mock_update.call_args)

    @patch("pytorch_agent.fixing_steps.implement.gh")
    @patch("pytorch_agent.fixing_steps.implement.update_stage")
    @patch("pytorch_agent.fixing_steps.implement.save_state")
    def test_escalation_after_max_attempts(self, mock_save, mock_update, mock_gh):
        tracked = self._make_tracked(attempt_count=3)
        run(tracked)
        mock_update.assert_called_once()
        assert "NEEDS_HUMAN" in str(mock_update.call_args)

    def test_skips_wrong_stage(self):
        tracked = self._make_tracked(stage="IN_REVIEW")
        run(tracked)  # Should return without error
