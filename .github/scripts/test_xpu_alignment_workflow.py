#!/usr/bin/env python3

import re
import unittest
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "workflows/xpu_alignment.yml"


class AlignmentWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")

    def test_manual_dispatch_only_exposes_scan_date(self) -> None:
        dispatch = self.text.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
        self.assertIn("scan_date:", dispatch)
        self.assertNotIn("runner:", dispatch)
        self.assertNotIn("auto_file:", dispatch)
        self.assertNotIn("notify:", dispatch)

    def test_xpu_jobs_use_the_dedicated_runner(self) -> None:
        self.assertEqual(self.text.count("runs-on: xpu-agent"), 2)
        self.assertNotIn("runs-on: ${{ inputs.runner", self.text)

    def test_agent_roles_surround_the_deterministic_runner(self) -> None:
        positions = [
            self.text.index("role `scan-prepare`"),
            self.text.index("xpu_alignment_runner.py"),
            self.text.index("role `scan-finalize`"),
            self.text.index("role `review`"),
            self.text.index("Build publishing decision"),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_only_gate_job_requests_issue_write_permission(self) -> None:
        self.assertEqual(self.text.count("issues: write"), 1)
        gate = self.text.split("  gate-and-publish:", 1)[1]
        self.assertRegex(gate, re.compile(r"permissions:\n(?:.*\n){0,4}\s+issues: write"))

    def test_schedule_and_dispatch_share_non_cancelling_concurrency(self) -> None:
        self.assertIn("group: xpu-alignment", self.text)
        self.assertIn("cancel-in-progress: false", self.text)
        self.assertIn("cron: '0 2 * * *'", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
