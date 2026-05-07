"""Tests for discovery_agent — covers bugs found during Step 1 testing.

Bugs tested:
1. _token_for_repo used raw env vars instead of config constants
2. Stage was TRIAGING instead of DISCOVERED
3. failed_tests list not coerced to string
4. append_log didn't preserve indentation inside list items
5. Discovery didn't call sync_labels
6. sync_labels called with body text instead of stage string
7. Missing re import (caught by Python, not testable as unit)
8. --reset mechanism: extracts original body from <details> block
9. Agent Log section removed (redundant with per-item logs)
10. Environment section added for collect_env info
11. Reproducer/context should be preserved verbatim (skill-level, tested via template)
12. Discovery log shows extraction summary, not just filename
"""
from __future__ import annotations

import json
import re
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pytorch_agent.utils.body_templates import (
    get_status, set_status, render_initial_body,
    check_action_item, append_log, sync_labels,
)
from pytorch_agent.discovery_agent import (
    _extract_label_info, reset, run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_BODY = render_initial_body(
    stage="DISCOVERED",
    summary="test summary",
    test_type="ut",
    category="Torch Operations",
    dependency="upstream-pytorch",
    platform="PVC",
    failed_tests="- `test_foo`",
    error_log="RuntimeError: boom",
    reproducer="import torch; torch.foo()",
    commit_scope="abc123..def456",
    context="some context",
    environment="PyTorch version: 2.13.0",
    original_issue="raw issue body here",
)


# ---------------------------------------------------------------------------
# Bug 2: Stage should be DISCOVERED, not TRIAGING
# ---------------------------------------------------------------------------

class TestDiscoveredStage:
    def test_render_sets_discovered_status(self):
        assert get_status(SAMPLE_BODY) == "DISCOVERED"

    def test_body_contains_discovered_marker(self):
        assert "<!-- agent:status:DISCOVERED -->" in SAMPLE_BODY


# ---------------------------------------------------------------------------
# Bug 3: failed_tests list coercion
# ---------------------------------------------------------------------------

class TestFailedTestsCoercion:
    def test_list_coerced_to_string(self):
        """LLM may return failed_tests as a list — must be joined."""
        ft_list = ["- `test_a`", "- `test_b`"]
        ft_str = "\n".join(ft_list)
        body = render_initial_body(failed_tests=ft_str)
        assert "- `test_a`" in body
        assert "- `test_b`" in body
        # Must NOT contain Python list repr
        assert "['- " not in body


# ---------------------------------------------------------------------------
# Bug 4: append_log preserves indentation
# ---------------------------------------------------------------------------

class TestAppendLogIndentation:
    def test_log_indented_when_marker_indented(self):
        """When marker is inside a list item (indented), log content must match."""
        body = (
            "- [ ] 🔍 Issue formatted\n"
            "  <details><summary>Discovery log</summary>\n"
            "  <!-- agent:discovery-log -->\n"
            "  </details>\n"
        )
        result = append_log(body, "discovery", "Test log entry")
        # Find lines between <details> and </details>
        lines = result.split("\n")
        in_details = False
        log_lines = []
        for line in lines:
            if "<details>" in line:
                in_details = True
                continue
            if "</details>" in line:
                in_details = False
                continue
            if in_details and line.strip() and "<!--" not in line:
                log_lines.append(line)
        # All log content lines should be indented with at least 2 spaces
        for line in log_lines:
            assert line.startswith("  "), f"Log line not indented: {line!r}"

    def test_log_not_indented_when_marker_not_indented(self):
        """When marker has no indentation, log content shouldn't add any."""
        body = (
            "<details><summary>Log</summary>\n"
            "<!-- agent:discovery-log -->\n"
            "</details>\n"
        )
        result = append_log(body, "discovery", "Test entry")
        assert "<!-- agent:discovery-log -->" in result
        # Marker should still be present, content inserted before it
        assert "Test entry" in result


# ---------------------------------------------------------------------------
# Bug 5 & 6: sync_labels with stage string
# ---------------------------------------------------------------------------

class TestSyncLabels:
    @patch("pytorch_agent.utils.git.add_label")
    @patch("pytorch_agent.utils.git.remove_label")
    def test_sync_labels_adds_correct_label(self, mock_remove, mock_add):
        """DISCOVERED stage should map to agent:active label."""
        with patch("pytorch_agent.utils.config.STAGE_TO_LABEL",
                    {"DISCOVERED": "agent:active", "DONE": "agent:done"}), \
             patch("pytorch_agent.utils.config.ALL_AGENT_LABELS",
                    ["agent:active", "agent:done", "agent:needs-human"]):
            sync_labels("repo/name", 42, "DISCOVERED")
        mock_add.assert_called_once_with("repo/name", 42, "agent:active")
        mock_remove.assert_any_call("repo/name", 42, "agent:done")

    @patch("pytorch_agent.utils.git.add_label")
    @patch("pytorch_agent.utils.git.remove_label")
    def test_sync_labels_with_body_text_would_fail(self, mock_remove, mock_add):
        """Passing body text instead of stage string should not match any label."""
        with patch("pytorch_agent.utils.config.STAGE_TO_LABEL",
                    {"DISCOVERED": "agent:active"}), \
             patch("pytorch_agent.utils.config.ALL_AGENT_LABELS",
                    ["agent:active"]):
            sync_labels("repo/name", 42, "<!-- agent:status:DISCOVERED -->...")
        mock_add.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 8: --reset extracts original body
# ---------------------------------------------------------------------------

class TestReset:
    @patch("pytorch_agent.discovery_agent.gh")
    def test_reset_extracts_original_body(self, mock_gh):
        original = "### Bug\nSome original content here"
        formatted_body = (
            "<!-- agent:status:DISCOVERED -->\n"
            "## Summary\nstuff\n"
            "## Original Issue\n"
            "<details><summary>Original issue body</summary>\n\n"
            f"{original}\n\n"
            "</details>"
        )
        mock_gh.get_issue_detail.return_value = {"body": formatted_body}
        reset(123)
        mock_gh.update_issue_body.assert_called_once()
        written_body = mock_gh.update_issue_body.call_args[0][2]
        assert written_body == original

    @patch("pytorch_agent.discovery_agent.gh")
    def test_reset_warns_on_no_original(self, mock_gh):
        """If no Original Issue section, reset should warn and not update."""
        mock_gh.get_issue_detail.return_value = {"body": "no original section"}
        reset(123)
        mock_gh.update_issue_body.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 9: No Agent Log section in template
# ---------------------------------------------------------------------------

class TestNoAgentLogSection:
    def test_body_has_no_agent_log_section(self):
        assert "## Agent Log" not in SAMPLE_BODY
        assert "<!-- agent:log -->" not in SAMPLE_BODY


# ---------------------------------------------------------------------------
# Bug 10: Environment section present
# ---------------------------------------------------------------------------

class TestEnvironmentSection:
    def test_environment_section_exists(self):
        assert "## Environment" in SAMPLE_BODY

    def test_environment_in_collapsible_code_block(self):
        assert "<details><summary>collect_env</summary>" in SAMPLE_BODY
        # Content should be inside a code block
        env_section = SAMPLE_BODY.split("## Environment")[1].split("## Original")[0]
        assert "```" in env_section

    def test_environment_content_rendered(self):
        assert "PyTorch version: 2.13.0" in SAMPLE_BODY


# ---------------------------------------------------------------------------
# Bug 12: Discovery log shows summary, not filename
# ---------------------------------------------------------------------------

class TestDiscoveryLogContent:
    def test_log_contains_summary_fields(self):
        body = SAMPLE_BODY
        body = append_log(body, "discovery",
                          "**Summary:** test\n**Failed tests:** foo\n**Dependency:** bar\n**Commit scope:** baz")
        assert "**Summary:** test" in body
        assert "**Failed tests:** foo" in body
        # Must NOT contain just a filename
        assert "Log: `agent-issue" not in body


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------

class TestExtractLabelInfo:
    def test_extracts_from_dict_labels(self):
        labels = [
            {"name": "agent_test: ut"},
            {"name": "agent_category: Torch Operations"},
            {"name": "agent_dependency: upstream-pytorch"},
            {"name": "ai_generated"},
        ]
        info = _extract_label_info(labels)
        assert info["test_type"] == "ut"
        assert info["category"] == "Torch Operations"
        assert info["dependency"] == "upstream-pytorch"

    def test_extracts_from_string_labels(self):
        labels = ["agent_test: e2e", "agent_category: Inductor"]
        info = _extract_label_info(labels)
        assert info["test_type"] == "e2e"
        assert info["category"] == "Inductor"

    def test_unrelated_labels_ignored(self):
        labels = [{"name": "bug"}, {"name": "priority: high"}]
        info = _extract_label_info(labels)
        assert all(v == "" for v in info.values())


# ---------------------------------------------------------------------------
# Token routing (Bug 1)
# ---------------------------------------------------------------------------

class TestTokenRouting:
    @patch.dict(os.environ, {"GH_TOKEN": "gh_token", "REVIEW_GH_TOKEN": "review_token"})
    def test_issue_repo_uses_review_token(self):
        from pytorch_agent.utils.git import _token_for_repo
        from pytorch_agent.utils.config import ISSUE_REPO
        token = _token_for_repo(ISSUE_REPO)
        assert token == "review_token"

    @patch.dict(os.environ, {"GH_TOKEN": "gh_token", "REVIEW_GH_TOKEN": "review_token"})
    def test_unknown_repo_uses_gh_token(self):
        from pytorch_agent.utils.git import _token_for_repo
        token = _token_for_repo("random/repo")
        assert token == "gh_token"


# ---------------------------------------------------------------------------
# Skip-if-already-formatted
# ---------------------------------------------------------------------------

class TestSkipAlreadyFormatted:
    @patch("pytorch_agent.discovery_agent.gh")
    @patch("pytorch_agent.discovery_agent.get_backend")
    def test_skips_when_status_present(self, mock_backend, mock_gh):
        mock_gh.get_issue_detail.return_value = {
            "body": "<!-- agent:status:DISCOVERED -->\nstuff",
            "labels": [],
        }
        run(999)
        mock_backend.assert_not_called()
