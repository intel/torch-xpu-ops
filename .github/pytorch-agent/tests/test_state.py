"""Tests for state management module."""
import json
import sys
import os
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.utils.state import (
    TrackedIssue, render_state_comment, parse_state_comment,
    save_state, update_stage, find_tracked_by_issue, load_tracked,
    STATE_COMMENT_MARKER,
)


class TestRenderParseRoundtrip:
    def test_basic_roundtrip(self):
        tracked = TrackedIssue(
            source_repo="intel/torch-xpu-ops",
            source_number=42,
            title="Fix XPU conv",
            stage="IMPLEMENTING",
            branch="agent/issue-42",
            review_iteration=1,
            attempt_count=2,
            last_push_sha="abc123",
        )
        body = render_state_comment(tracked)
        parsed = parse_state_comment(body)
        assert parsed is not None
        assert parsed.source_number == 42
        assert parsed.stage == "IMPLEMENTING"
        assert parsed.branch == "agent/issue-42"
        assert parsed.review_iteration == 1
        assert parsed.attempt_count == 2
        assert parsed.last_push_sha == "abc123"

    def test_roundtrip_all_fields(self):
        tracked = TrackedIssue(
            source_repo="intel/torch-xpu-ops",
            source_number=99,
            title="Test",
            stage="IN_REVIEW",
            tracking_pr_number=5,
            tracking_pr_url="https://github.com/test-review/pytorch/pull/5",
            public_pr_number=1234,
            public_pr_url="https://github.com/pytorch/pytorch/pull/1234",
            branch="agent/issue-99",
            triage_reason="pytorch fix needed",
            review_iteration=3,
            attempt_count=1,
            last_push_sha="def456",
        )
        body = render_state_comment(tracked)
        parsed = parse_state_comment(body)
        assert parsed.tracking_pr_number == 5
        assert parsed.public_pr_number == 1234
        assert parsed.triage_reason == "pytorch fix needed"

    def test_parse_returns_none_for_no_marker(self):
        assert parse_state_comment("just a regular comment") is None

    def test_render_contains_marker(self):
        tracked = TrackedIssue("r", 1, "t")
        body = render_state_comment(tracked)
        assert STATE_COMMENT_MARKER in body
        assert "DISCOVERED" in body


class TestStateOperations:
    @patch("pytorch_agent.utils.state.gh")
    def test_save_state_creates_comment(self, mock_gh):
        mock_gh.get_issue_comments.return_value = []
        tracked = TrackedIssue("intel/torch-xpu-ops", 42, "Fix")
        tracked._state_comment_id = None

        # After add_issue_comment, _find_state_comment is called
        # Simulate: first call returns empty (for save), second returns the comment
        state_body = render_state_comment(tracked)
        mock_gh.get_issue_comments.side_effect = [
            [{"id": 100, "body": state_body}]  # re-fetch
        ]

        save_state(tracked)
        mock_gh.add_issue_comment.assert_called_once()
        mock_gh.add_label.assert_called()  # tracking + stage label

    @patch("pytorch_agent.utils.state.gh")
    def test_save_state_updates_existing(self, mock_gh):
        tracked = TrackedIssue("intel/torch-xpu-ops", 42, "Fix")
        tracked._state_comment_id = 100

        save_state(tracked)
        mock_gh.update_issue_comment.assert_called_once_with(
            "intel/torch-xpu-ops", 100, render_state_comment(tracked)
        )

    @patch("pytorch_agent.utils.state.gh")
    @patch("pytorch_agent.utils.state.log")
    def test_update_stage(self, mock_log, mock_gh):
        tracked = TrackedIssue("intel/torch-xpu-ops", 42, "Fix")
        tracked._state_comment_id = 100

        update_stage(tracked, "IMPLEMENTING", "Triage complete")
        assert tracked.stage == "IMPLEMENTING"
        # Should post human-readable comment
        assert mock_gh.add_issue_comment.call_count >= 1

    @patch("pytorch_agent.utils.state.gh")
    def test_load_tracked_raises_if_not_found(self, mock_gh):
        mock_gh.get_issue_comments.return_value = []
        with pytest.raises(ValueError, match="No agent state"):
            load_tracked(999)
