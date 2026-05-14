"""Tests for verify_fix module."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from issue_handler.verify_fix import (
    run,
    _needs_rebuild,
    _sync_to_pytorch,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestNeedsRebuild:
    def test_cpp_file(self):
        assert _needs_rebuild("src/ATen/native/xpu/SummaryOps.cpp\n")

    def test_python_only(self):
        assert not _needs_rebuild("test/xpu/test_ops.py\n")

    def test_header_file(self):
        assert _needs_rebuild("src/ATen/native/xpu/sycl/Foo.h\n")

    def test_mixed(self):
        assert _needs_rebuild("test/xpu/test_ops.py\nsrc/Foo.cpp\n")

    def test_empty(self):
        assert not _needs_rebuild("")

    def test_sycl_file(self):
        assert _needs_rebuild("src/kernel.sycl\n")


# ---------------------------------------------------------------------------
# Integration tests (mocked)
# ---------------------------------------------------------------------------

def _make_issue_body(*, status="IN_REVIEW", target="torch-xpu-ops",
                     tracking_pr="3670", verified=False,
                     test_cmd="pytest -v test_ops.py"):
    """Build a minimal issue body for testing."""
    verified_check = "[x]" if verified else "[ ]"
    return (
        f"<!-- agent:status:{status} -->\n"
        f"<!-- target_repo: {target} -->\n"
        f"<!-- tracking_pr: #{tracking_pr} -->\n"
        "## Failed Tests\n"
        "- `third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestFoo::test_bar_xpu_float32`\n\n"
        "## Reproducer\n"
        f"```bash\n{test_cmd}\n```\n\n"
        "## Action Items\n"
        "- [x] 🔍 Issue formatted (Discovery Agent)\n"
        "- [x] 🧠 Root cause identified (Triage Agent)\n"
        "- [x] 🔧 Fix implemented (Fix Agent)\n"
        f"- {verified_check} ✅ Fix verified locally\n"
        "- [x] 📋 PR proposed\n"
        "- [ ] 👀 Human review\n"
        "- [ ] 🎉 PR merged\n"
    )


class TestVerifyFixRun:
    """Test the run() function with mocked GitHub API and git commands."""

    @patch("issue_handler.verify_fix.gh")
    def test_skip_non_in_review(self, mock_gh):
        """Should skip if issue is not at IN_REVIEW stage."""
        mock_gh.get_issue_detail.return_value = {
            "body": _make_issue_body(status="IMPLEMENTING"),
        }
        result = run(1234)
        assert result is False

    @patch("issue_handler.verify_fix.gh")
    def test_skip_already_verified(self, mock_gh):
        """Should skip if already verified."""
        mock_gh.get_issue_detail.return_value = {
            "body": _make_issue_body(verified=True),
        }
        result = run(1234)
        assert result is True

    @patch("issue_handler.verify_fix.gh")
    def test_no_tracking_pr(self, mock_gh):
        """Should return False if no tracking PR."""
        mock_gh.get_issue_detail.return_value = {
            "body": _make_issue_body(tracking_pr=""),
        }
        result = run(1234)
        assert result is False

    @patch("issue_handler.verify_fix._restore_xpu_txt")
    @patch("issue_handler.verify_fix._cleanup_xpu_ops_worktree")
    @patch("issue_handler.verify_fix._checkout_xpu_ops_pr")
    @patch("issue_handler.verify_fix._sync_to_pytorch")
    @patch("issue_handler.verify_fix._rebuild_pytorch")
    @patch("issue_handler.verify_fix._run_test")
    @patch("issue_handler.verify_fix.git_out")
    @patch("issue_handler.verify_fix.git")
    @patch("issue_handler.verify_fix.gh")
    def test_torch_xpu_ops_pass(self, mock_gh, mock_git, mock_git_out,
                                 mock_run_test, mock_rebuild, mock_sync,
                                 mock_checkout, mock_cleanup, mock_restore):
        """Should verify successfully for torch-xpu-ops target."""
        mock_gh.get_issue_detail.return_value = {
            "body": _make_issue_body(
                test_cmd="pytest -xvs third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestFoo"),
        }
        mock_checkout.return_value = (Path("/tmp/verify-1234"), "main")
        mock_git_out.return_value = "src/ATen/native/xpu/Foo.py\n"
        mock_rebuild.return_value = (True, "Build OK")
        mock_run_test.return_value = (True, "1 passed\n")

        result = run(1234)

        assert result is True
        # Should have updated the issue body with verified checkbox
        mock_gh.update_issue_body.assert_called_once()
        updated_body = mock_gh.update_issue_body.call_args[0][2]
        assert "[x]" in updated_body and "Fix verified locally" in updated_body
        # Should have posted a success comment
        mock_gh.add_issue_comment.assert_called_once()
        comment = mock_gh.add_issue_comment.call_args[0][2]
        assert "verification passed" in comment.lower()

    @patch("issue_handler.verify_fix._restore_xpu_txt")
    @patch("issue_handler.verify_fix._cleanup_xpu_ops_worktree")
    @patch("issue_handler.verify_fix._checkout_xpu_ops_pr")
    @patch("issue_handler.verify_fix._sync_to_pytorch")
    @patch("issue_handler.verify_fix._rebuild_pytorch")
    @patch("issue_handler.verify_fix._run_test")
    @patch("issue_handler.verify_fix.git_out")
    @patch("issue_handler.verify_fix.git")
    @patch("issue_handler.verify_fix.gh")
    def test_torch_xpu_ops_fail(self, mock_gh, mock_git, mock_git_out,
                                 mock_run_test, mock_rebuild, mock_sync,
                                 mock_checkout, mock_cleanup, mock_restore):
        """Should move to NEEDS_HUMAN on verification failure."""
        mock_gh.get_issue_detail.return_value = {
            "body": _make_issue_body(
                test_cmd="pytest -xvs third_party/torch-xpu-ops/test/xpu/test_ops_xpu.py::TestFoo"),
        }
        mock_checkout.return_value = (Path("/tmp/verify-1234"), "main")
        mock_git_out.return_value = "src/ATen/native/xpu/Foo.py\n"
        mock_rebuild.return_value = (True, "Build OK")
        mock_run_test.return_value = (False, "FAILED test_bar\n")

        result = run(1234)

        assert result is False
        # Should have set status to NEEDS_HUMAN
        mock_gh.update_issue_body.assert_called_once()
        updated_body = mock_gh.update_issue_body.call_args[0][2]
        assert "NEEDS_HUMAN" in updated_body
        # Should have posted a failure comment
        mock_gh.add_issue_comment.assert_called_once()
        comment = mock_gh.add_issue_comment.call_args[0][2]
        assert "verification failed" in comment.lower()

    @patch("issue_handler.verify_fix.git")
    @patch("issue_handler.verify_fix.gh")
    def test_pytorch_target_pass(self, mock_gh, mock_git):
        """Should verify successfully for pytorch target."""
        body = _make_issue_body(
            target="pytorch",
            test_cmd="pytest -xvs test/test_torch.py::TestFoo",
        )
        mock_gh.get_issue_detail.return_value = {"body": body}

        with patch("issue_handler.verify_fix.git_out") as mock_git_out, \
             patch("issue_handler.verify_fix._run_test") as mock_run_test:
            mock_git_out.return_value = "torch/nn/modules/module.py\n"
            mock_run_test.return_value = (True, "1 passed\n")

            result = run(1234)

        assert result is True
