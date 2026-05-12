"""Tests for e2e_report and TokenUsage."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from issue_handler.utils.agent_backend import TokenUsage
from e2e_report import parse_issue_status, build_report


class TestTokenUsage:
    def test_defaults(self):
        t = TokenUsage()
        assert t.total == 0
        assert t.cost == 0.0

    def test_add_step(self):
        t = TokenUsage()
        t.add_step({"total": 1000, "input": 500, "output": 200,
                     "cache": {"read": 200, "write": 100}})
        assert t.total == 1000
        assert t.input == 500
        assert t.cache_read == 200

    def test_add_multiple_steps(self):
        t = TokenUsage()
        t.add_step({"total": 1000, "input": 500, "output": 200, "cache": {}})
        t.add_step({"total": 2000, "input": 1000, "output": 400, "cache": {"read": 500}})
        assert t.total == 3000
        assert t.input == 1500
        assert t.cache_read == 500

    def test_summary_formatting(self):
        t = TokenUsage(input=17000, output=1500, total=37000,
                       cache_read=17500, cache_write=0, cost=0.05)
        s = t.summary()
        assert "37.0K" in s
        assert "17.0K" in s
        assert "1.5K" in s
        assert "$0.0500" in s
        assert "model:" in s

    def test_summary_millions(self):
        t = TokenUsage(total=1_500_000, input=1_000_000, output=500_000)
        s = t.summary()
        assert "1.5M" in s


class TestParseIssueStatus:
    def test_unformatted(self):
        body = "- [ ] 🔍 Issue formatted\n- [ ] 🧠 Root cause identified"
        result = parse_issue_status(body)
        assert result["stage"] == "unformatted"
        assert result["stages_done"] == []

    def test_formatted_only(self):
        body = "- [x] 🔍 Issue formatted\n- [ ] 🧠 Root cause identified"
        result = parse_issue_status(body)
        assert result["stage"] == "formatted"
        assert "formatted" in result["stages_done"]

    def test_triaged(self):
        body = ("- [x] 🔍 Issue formatted\n"
                "- [x] 🧠 Root cause identified\n"
                "- [ ] 🔧 Fix implemented")
        result = parse_issue_status(body)
        assert result["stage"] == "triaged"

    def test_needs_human_failure(self):
        body = ("- [x] 🔍 Issue formatted\n"
                "- [x] 🧠 Root cause identified\n"
                "**Verdict:** NEEDS_HUMAN\n"
                "**Reason:** Upstream PyTorch change needed\n")
        result = parse_issue_status(body)
        assert result["failure_reason"] == "Upstream PyTorch change needed"
        assert result["failure_category"] == "upstream_dependency"

    def test_token_extraction(self):
        body = ("<details><summary>discovery log</summary>\n"
                "some stuff\n"
                "**Tokens:** tokens: 37.0K | in: 17.0K | out: 1.5K\n"
                "<!-- agent:discovery-log -->\n"
                "</details>\n")
        result = parse_issue_status(body)
        assert "format" in result["tokens"]
        assert "37.0K" in result["tokens"]["format"]


class TestBuildReport:
    def test_builds_markdown_table(self):
        results = [{
            "number": 123,
            "title": "Test issue",
            "stage": "formatted",
            "stages_done": ["formatted"],
            "tokens": {"format": "tokens: 37.0K"},
            "failure_reason": None,
            "failure_category": None,
            "state": "open",
        }]
        report = build_report(results)
        assert "# E2E Pipeline Test Dashboard" in report
        assert "#123" in report
        assert "IN PROGRESS" in report
        assert "Cost" in report
        assert "Model" in report

    def test_build_report_merges_existing_rows(self):
        """Rows from previous batches are preserved when building a new report."""
        existing_body = (
            "# E2E Pipeline Test Dashboard\n\n"
            "| # | Title | Format | Triage | Fix | PR | Review | Model | Tokens (fmt/tri/fix) | Cost | Result | Failure Reason |\n"
            "|---|-------|--------|--------|-----|----|--------|-------|---------------------|------|--------|----------------|\n"
            "| [#99](https://github.com/test/repo/issues/99) | Old issue | ✅ | ✅ | ✅ | ✅ | ⬜ | claude-sonnet-4 | 10K / 20K / 30K | $0.50 | 🔄 IN PROGRESS | — |\n"
        )
        new_results = [{
            "number": 200,
            "title": "New issue",
            "stage": "formatted",
            "stages_done": ["formatted"],
            "tokens": {"format": "tokens: 5K"},
            "failure_reason": None,
            "failure_category": None,
            "state": "open",
        }]
        report = build_report(new_results, existing_body=existing_body)
        # New issue present
        assert "#200" in report
        # Old issue preserved
        assert "#99" in report
        assert "Old issue" in report
        # Total includes both
        assert "Total:** 2" in report
