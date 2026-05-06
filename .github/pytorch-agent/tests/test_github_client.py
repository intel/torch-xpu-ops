"""Tests for github_client module."""
import json
import subprocess
from unittest.mock import patch, MagicMock, call
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.utils.github_client import (
    _gh, _gh_api, get_issues, get_issue_detail, add_issue_comment,
    close_issue, add_label, remove_label, create_draft_pr,
    get_pr_reviews, get_pr_status, get_ci_checks, delete_branch,
    create_cross_fork_pr, list_prs,
)


@pytest.fixture
def mock_subprocess():
    with patch("pytorch_agent.utils.github_client.subprocess.run") as mock:
        yield mock


class TestGhHelpers:
    def test_gh_runs_command(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout="output", returncode=0)
        result = _gh(["issue", "list"])
        mock_subprocess.assert_called_once()
        cmd = mock_subprocess.call_args[0][0]
        assert cmd == ["gh", "issue", "list"]
        assert result == "output"

    def test_gh_api_string_fields_use_f(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout='{"ok": true}', returncode=0)
        _gh_api("/repos/test/issues", method="POST", title="hello")
        cmd = mock_subprocess.call_args[0][0]
        assert "-f" in cmd
        idx = cmd.index("-f")
        assert cmd[idx + 1] == "title=hello"

    def test_gh_api_non_string_fields_use_F(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout='{"ok": true}', returncode=0)
        _gh_api("/repos/test/issues", method="POST", draft=True)
        cmd = mock_subprocess.call_args[0][0]
        assert "-F" in cmd

    def test_gh_api_empty_response(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        result = _gh_api("/repos/test/issues")
        assert result == {}


class TestIssues:
    def test_get_issues(self, mock_subprocess):
        issues = [{"number": 1, "title": "test"}]
        mock_subprocess.return_value = MagicMock(stdout=json.dumps(issues), returncode=0)
        result = get_issues("intel/torch-xpu-ops", "ai_generated")
        assert result == issues

    def test_add_issue_comment(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        add_issue_comment("intel/torch-xpu-ops", 42, "hello")
        cmd = mock_subprocess.call_args[0][0]
        assert "comment" in cmd
        assert "42" in cmd

    def test_close_issue(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        close_issue("intel/torch-xpu-ops", 42)
        cmd = mock_subprocess.call_args[0][0]
        assert "close" in cmd

    def test_add_label(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        add_label("intel/torch-xpu-ops", 42, "agent:tracking")
        cmd = mock_subprocess.call_args[0][0]
        assert "--add-label" in cmd

    def test_remove_label(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout="", returncode=0)
        remove_label("intel/torch-xpu-ops", 42, "agent:tracking")
        cmd = mock_subprocess.call_args[0][0]
        assert "--remove-label" in cmd


class TestPullRequests:
    def test_get_pr_status_merged(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout=json.dumps({"state": "closed", "merged": True}), returncode=0)
        assert get_pr_status("repo", 1) == "merged"

    def test_get_pr_status_open(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout=json.dumps({"state": "open", "merged": False}), returncode=0)
        assert get_pr_status("repo", 1) == "open"

    def test_create_cross_fork_pr(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(stdout=json.dumps({"number": 99, "url": "https://..."}), returncode=0)
        result = create_cross_fork_pr(
            "test-review/pytorch", "agent/issue-5",
            "pytorch/pytorch", "Fix XPU", "body"
        )
        assert result["number"] == 99


class TestCI:
    def test_get_ci_checks(self, mock_subprocess):
        # First call: get PR (for sha), second call: get checks
        pr_response = json.dumps({"head": {"sha": "abc123"}})
        checks_response = json.dumps({"check_runs": [{"name": "test", "conclusion": "success"}]})
        mock_subprocess.side_effect = [
            MagicMock(stdout=pr_response, returncode=0),
            MagicMock(stdout=checks_response, returncode=0),
        ]
        result = get_ci_checks("repo", 1)
        assert len(result) == 1
        assert result[0]["name"] == "test"
