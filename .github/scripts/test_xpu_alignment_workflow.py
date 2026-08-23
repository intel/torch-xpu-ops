#!/usr/bin/env python3

import re
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "workflows/xpu_alignment.yml"
BOT_WORKFLOW = Path(__file__).parents[1] / "workflows/bot.yml"


class AlignmentWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.bot_text = BOT_WORKFLOW.read_text(encoding="utf-8")

    def test_manual_dispatch_only_exposes_scan_date(self) -> None:
        dispatch = self.text.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
        self.assertIn("scan_date:", dispatch)
        self.assertNotIn("runner:", dispatch)
        self.assertNotIn("auto_file:", dispatch)
        self.assertNotIn("notify:", dispatch)

    def test_xpu_jobs_use_the_dedicated_runner(self) -> None:
        self.assertEqual(self.text.count("runs-on: xpu-agent"), 2)
        self.assertNotIn("runs-on: ${{ inputs.runner", self.text)

    def test_alignment_uses_the_bedrock_model_allowed_by_bot_policy(self) -> None:
        allowed = "BEDROCK_MODEL: us.anthropic.claude-opus-4-5-20251101-v1:0"
        self.assertIn(allowed, self.text)
        self.assertIn(allowed, self.bot_text)
        self.assertNotIn("BEDROCK_MODEL: global.", self.text)
        self.assertNotIn("BEDROCK_MODEL: global.", self.bot_text)

    def test_agent_roles_surround_the_deterministic_runner(self) -> None:
        positions = [
            self.text.index("role `scan-prepare`"),
            self.text.index("xpu_alignment_runner.py"),
            self.text.index("role `scan-finalize`"),
            self.text.index("role `review`"),
            self.text.index("Build publishing decision"),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_untrusted_reproducer_user_has_no_outbound_network(self) -> None:
        runner = self.text.split("  run-reproducers:", 1)[1].split(
            "  scan-finalize:", 1
        )[0]
        self.assertIn("--cap-add=NET_ADMIN", runner)
        self.assertIn("iptables --wait --append OUTPUT", runner)
        self.assertIn("ip6tables --wait --append OUTPUT", runner)
        self.assertIn('--uid-owner "$repro_uid"', runner)
        self.assertIn("Outbound network remains available to xpu-repro", runner)

    def test_only_gate_job_requests_issue_write_permission(self) -> None:
        self.assertEqual(self.text.count("issues: write"), 1)
        gate = self.text.split("  gate-and-publish:", 1)[1]
        self.assertRegex(gate, re.compile(r"permissions:\n(?:.*\n){0,4}\s+issues: write"))

    def test_schedule_and_dispatch_share_non_cancelling_concurrency(self) -> None:
        self.assertIn("group: xpu-alignment", self.text)
        self.assertIn("cancel-in-progress: false", self.text)
        self.assertIn("cron: '0 2 * * *'", self.text)

    def test_automatic_and_manual_filing_share_one_lock(self) -> None:
        gate = self.text.split("  gate-and-publish:", 1)[1]
        file_job = self.bot_text.split("  file:", 1)[1].split("  triage:", 1)[0]
        for job in (gate, file_job):
            self.assertIn("group: xpu-alignment-filing", job)
            self.assertIn("cancel-in-progress: false", job)


if __name__ == "__main__":
    unittest.main(verbosity=2)
