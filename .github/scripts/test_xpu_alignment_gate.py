#!/usr/bin/env python3
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from xpu_alignment_gate import build_decision


class AlignmentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scan_root = self.root / "scan"
        self.review_root = self.root / "reviewed"
        self.run = self.scan_root / "runs/2026-08-20"
        self.review_run = self.review_root / "runs/2026-08-20"
        (self.run / "scripts").mkdir(parents=True)
        (self.run / "logs").mkdir()
        (self.review_run / "review").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_run(self) -> tuple[Path, Path]:
        (self.run / "scripts/repro_issue-123.py").write_text("print('xpu')\n")
        (self.run / "logs/issue-123.log").write_text("observed upstream failure\n")
        scan = {
            "schema_version": 1,
            "status": "complete",
            "scan_window": {
                "start": "2026-08-20T00:00:00Z",
                "end": "2026-08-21T00:00:00Z",
            },
            "collection": {
                "complete": True,
                "sources": [
                    "issues-created",
                    "prs-created",
                    "prs-merged",
                    "default-branch-commits",
                ],
                "errors": [],
            },
            "environment": {"xpu_available": True},
            "candidates": [
                {
                    "id": "issue-123",
                    "kind": "issue",
                    "title": "Upstream failure",
                    "url": "https://github.com/pytorch/pytorch/issues/123",
                    "triage": "validate",
                    "reason": "same operator path",
                    "local_result": "confirmed",
                    "reproducer": "scripts/repro_issue-123.py",
                    "log": "logs/issue-123.log",
                    "target_path_verified": True,
                    "oracle": "raises RuntimeError",
                }
            ],
            "blockers": [],
        }
        scan_path = self.run / "scan.json"
        scan_path.write_text(json.dumps(scan) + "\n")
        review = {
            "schema_version": 1,
            "status": "complete",
            "scan_sha256": hashlib.sha256(scan_path.read_bytes()).hexdigest(),
            "units": [
                {
                    "id": "issue-123",
                    "verdict": "needs-xpu-fix",
                    "implementation_repository": "intel/torch-xpu-ops",
                    "canonical_tracker": None,
                    "payload": {
                        "title": "[xpu-alignment] Fix upstream failure on XPU",
                        "body": "Source and reproducer evidence.",
                        "labels": ["ai_generated"],
                    },
                }
            ],
            "blockers": [],
        }
        review_path = self.review_run / "review/review.json"
        review_path.write_text(json.dumps(review) + "\n")
        return scan_path, review_path

    def decision(self, *, auto_file: bool = True, producers_clean: bool = True) -> dict:
        return build_decision(
            self.scan_root,
            self.review_root,
            auto_file=auto_file,
            producers_clean=producers_clean,
            run_id="42",
            scan_date="2026-08-20",
        )

    def test_one_reviewed_candidate_is_filed_automatically(self) -> None:
        self.write_run()

        decision = self.decision()

        self.assertEqual(decision["decision"], "file-one")
        self.assertEqual(decision["actionable_units"], ["issue-123"])
        self.assertEqual(decision["payloads"][0]["unit_id"], "issue-123")

    def test_manual_run_without_auto_file_routes_candidate_to_triage(self) -> None:
        self.write_run()

        decision = self.decision(auto_file=False)

        self.assertEqual(decision["decision"], "triage")

    def test_incomplete_review_coverage_blocks_publishing(self) -> None:
        _, review_path = self.write_run()
        review = json.loads(review_path.read_text())
        review["units"] = []
        review_path.write_text(json.dumps(review) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertIn("review-coverage-mismatch", decision["blockers"])
        self.assertEqual(decision["payloads"], [])

    def test_review_of_different_scan_bytes_blocks_publishing(self) -> None:
        _, review_path = self.write_run()
        review = json.loads(review_path.read_text())
        review["scan_sha256"] = "0" * 64
        review_path.write_text(json.dumps(review) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertIn("review-scan-digest-mismatch", decision["blockers"])

    def test_two_reviewed_candidates_always_require_human_triage(self) -> None:
        scan_path, review_path = self.write_run()
        scan = json.loads(scan_path.read_text())
        second = dict(scan["candidates"][0])
        second.update(
            {
                "id": "pr-456",
                "kind": "pr",
                "url": "https://github.com/pytorch/pytorch/pull/456",
                "reproducer": "scripts/repro_pr-456.py",
                "log": "logs/pr-456.log",
            }
        )
        scan["candidates"].append(second)
        (self.run / second["reproducer"]).write_text("print('second')\n")
        (self.run / second["log"]).write_text("second failure\n")
        scan_path.write_text(json.dumps(scan) + "\n")

        review = json.loads(review_path.read_text())
        second_unit = json.loads(json.dumps(review["units"][0]))
        second_unit["id"] = "pr-456"
        second_unit["payload"]["title"] = "[xpu-alignment] Fix second failure"
        review["units"].append(second_unit)
        review["scan_sha256"] = hashlib.sha256(scan_path.read_bytes()).hexdigest()
        review_path.write_text(json.dumps(review) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "triage")
        self.assertEqual(decision["actionable_units"], ["issue-123", "pr-456"])

    def test_failed_producer_cannot_unlock_unattended_filing(self) -> None:
        self.write_run()

        decision = self.decision(producers_clean=False)

        self.assertEqual(decision["decision"], "triage")
        self.assertTrue(decision["needs_attention"])
        self.assertIn("producer-job-failed", decision["attention_reasons"])

    def test_complete_quiet_day_publishes_nothing(self) -> None:
        scan_path, review_path = self.write_run()
        scan = json.loads(scan_path.read_text())
        scan["candidates"] = []
        scan_path.write_text(json.dumps(scan) + "\n")
        review = json.loads(review_path.read_text())
        review["units"] = []
        review["scan_sha256"] = hashlib.sha256(scan_path.read_bytes()).hexdigest()
        review_path.write_text(json.dumps(review) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "none")
        self.assertFalse(decision["needs_attention"])

    def test_evidence_path_cannot_escape_the_scan_run(self) -> None:
        scan_path, review_path = self.write_run()
        scan = json.loads(scan_path.read_text())
        scan["candidates"][0]["log"] = "../../outside.log"
        scan_path.write_text(json.dumps(scan) + "\n")
        review = json.loads(review_path.read_text())
        review["scan_sha256"] = hashlib.sha256(scan_path.read_bytes()).hexdigest()
        review_path.write_text(json.dumps(review) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertIn("log:issue-123-path-outside-run", decision["blockers"])

    def test_verification_gap_notifies_maintainers_without_publishing(self) -> None:
        _, review_path = self.write_run()
        review = json.loads(review_path.read_text())
        review["units"][0]["verdict"] = "verification-gap"
        review["units"][0]["payload"] = None
        review_path.write_text(json.dumps(review) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "none")
        self.assertTrue(decision["needs_attention"])
        self.assertIn("review-verification-gap", decision["attention_reasons"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
