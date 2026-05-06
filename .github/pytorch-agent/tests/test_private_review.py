"""Tests for fixing_steps/private_review.py."""
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.utils.state import TrackedIssue
from pytorch_agent.fixing_steps.private_review import run


class TestPrivateReview:
    def _make_tracked(self, **kwargs):
        defaults = dict(
            source_repo="intel/torch-xpu-ops",
            source_number=42,
            title="Fix XPU conv",
            stage="IN_REVIEW",
            tracking_pr_number=5,
            branch="agent/issue-42",
            review_iteration=0,
            _state_comment_id=100,
        )
        defaults.update(kwargs)
        return TrackedIssue(**defaults)

    @patch("pytorch_agent.fixing_steps.private_review.get_review_state")
    @patch("pytorch_agent.fixing_steps.private_review.update_stage")
    def test_approved_advances(self, mock_update, mock_state):
        mock_state.return_value = "approved"
        tracked = self._make_tracked()
        run(tracked)
        mock_update.assert_called_once()
        assert "PUBLIC_PR" in str(mock_update.call_args)

    @patch("pytorch_agent.fixing_steps.private_review.get_review_state")
    def test_pending_noop(self, mock_state):
        mock_state.return_value = "pending"
        tracked = self._make_tracked()
        run(tracked)  # Should not raise

    @patch("pytorch_agent.fixing_steps.private_review.gh")
    @patch("pytorch_agent.fixing_steps.private_review.get_review_state")
    @patch("pytorch_agent.fixing_steps.private_review.update_stage")
    @patch("pytorch_agent.fixing_steps.private_review.save_state")
    def test_escalation_after_max_iterations(self, mock_save, mock_update, mock_state, mock_gh):
        mock_state.return_value = "changes_requested"
        tracked = self._make_tracked(review_iteration=3)
        run(tracked)
        mock_update.assert_called_once()
        assert "NEEDS_HUMAN" in str(mock_update.call_args)

    @patch("pytorch_agent.fixing_steps.private_review.gh")
    @patch("pytorch_agent.fixing_steps.private_review.subprocess")
    @patch("pytorch_agent.fixing_steps.private_review.get_backend")
    @patch("pytorch_agent.fixing_steps.private_review.get_pending_reviews")
    @patch("pytorch_agent.fixing_steps.private_review.format_reviews_for_prompt")
    @patch("pytorch_agent.fixing_steps.private_review.get_review_state")
    @patch("pytorch_agent.fixing_steps.private_review.save_state")
    def test_changes_requested_dispatches_agent(self, mock_save, mock_state,
                                                 mock_format, mock_pending,
                                                 mock_backend, mock_subproc,
                                                 mock_gh):
        mock_state.return_value = "changes_requested"
        mock_pending.return_value = [{"body": "fix this", "user": {"login": "reviewer1"}}]
        mock_format.return_value = "review text"
        mock_backend_inst = MagicMock()
        mock_backend_inst.run.return_value = "done"
        mock_backend.return_value = mock_backend_inst
        mock_subproc.run.return_value = MagicMock(stdout="sha123", returncode=0)
        mock_gh.add_pr_comment.return_value = {"id": 42}

        tracked = self._make_tracked()
        run(tracked)

        assert tracked.review_iteration == 1
        mock_backend_inst.run.assert_called_once()
        # Verify task list comment was posted and updated
        mock_gh.add_pr_comment.assert_called_once()
        mock_gh.update_pr_comment.assert_called_once()
        # Final comment should have all tasks checked and @mention reviewer
        final_body = mock_gh.update_pr_comment.call_args[1].get("body") or mock_gh.update_pr_comment.call_args[0][2]
        assert "[x]" in final_body
        assert "@reviewer1" in final_body
