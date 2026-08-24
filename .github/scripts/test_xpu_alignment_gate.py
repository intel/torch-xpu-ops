#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from xpu_alignment_collect import collect
from xpu_alignment_gate import build_decision


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AlignmentArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.collection_root = root / "collection"
        self.prepare_root = root / "prepare"
        self.runner_root = root / "runner"
        self.scan_root = root / "scan"
        self.review_root = root / "review"
        for path in (
            self.collection_root,
            self.prepare_root,
            self.runner_root,
            self.scan_root,
            self.review_root,
        ):
            path.mkdir(parents=True)

    def write(self, unit_ids: list[str] | None = None) -> dict[str, Path]:
        unit_ids = ["issue-123"] if unit_ids is None else unit_ids
        scan_window = {
            "start": "2026-08-20T00:00:00Z",
            "end": "2026-08-21T00:00:00Z",
        }

        class GitHub:
            def snapshot(self, repository: str) -> dict:
                return {"default_branch": "main", "default_branch_head": "a" * 40}

            def page(self, repository, source, cursor, snapshot, window):
                nodes = []
                if source == "issues-created":
                    nodes = [
                        {
                            "id": unit_id,
                            "kind": "issue",
                            "title": f"Candidate {unit_id}",
                            "url": (
                                "https://github.com/pytorch/pytorch/issues/"
                                f"{unit_id.rsplit('-', 1)[-1]}"
                            ),
                            "event_at": "2026-08-20T01:00:00Z",
                        }
                        for unit_id in unit_ids
                    ]
                return {
                    "nodes": nodes,
                    "page_info": {"has_next_page": False, "end_cursor": None},
                    "rate": {"remaining": 900, "reset_at": "2026-08-21T03:00:00Z"},
                    "raw": {"nodes": nodes},
                }

        collection = collect("pytorch/pytorch", scan_window, self.collection_root, GitHub())
        collection_path = self.collection_root / "collection/collection.json"
        scripts = self.prepare_root / "scripts"
        logs = self.runner_root / "runner/logs"
        reviews = self.review_root / "review"
        scripts.mkdir()
        logs.mkdir(parents=True)
        reviews.mkdir()
        decisions = []
        executions = []
        results = []
        candidates = []
        units = []
        for unit_id in unit_ids:
            number = unit_id.rsplit("-", 1)[-1]
            script = scripts / f"repro_{unit_id}.py"
            script.write_text("print('xpu evidence')\n")
            log = logs / f"{unit_id}.log"
            log.write_text("target XPU path raised the upstream error\n")
            decisions.append(
                {
                    "id": unit_id,
                    "triage": "validate",
                    "reason": "shared operator path",
                }
            )
            executions.append(
                {
                    "id": unit_id,
                    "script": f"scripts/repro_{unit_id}.py",
                    "script_sha256": digest(script),
                    "timeout_seconds": 120,
                    "oracle": "raises RuntimeError",
                    "target_path": "ATen operator",
                }
            )
            results.append(
                {
                    "id": unit_id,
                    "script_sha256": digest(script),
                    "command": ["/usr/bin/python3", "-I", f"scripts/repro_{unit_id}.py"],
                    "log": f"runner/logs/{unit_id}.log",
                    "log_sha256": digest(log),
                    "returncode": 0,
                    "timed_out": False,
                    "duration_seconds": 0.1,
                    "error": None,
                }
            )
            candidates.append(
                {
                    "id": unit_id,
                    "local_result": "confirmed",
                    "target_path_verified": True,
                    "evidence": f"runner/logs/{unit_id}.log",
                }
            )
            units.append(
                {
                    "id": unit_id,
                    "verdict": "needs-xpu-fix",
                    "implementation_repository": "intel/torch-xpu-ops",
                    "canonical_tracker": None,
                    "payload": {
                        "title": f"[xpu-alignment] Fix {unit_id} on XPU",
                        "body": "Reviewed source and runner evidence.",
                        "labels": ["ai_generated"],
                    },
                }
            )
        prepare = {
            "schema_version": 1,
            "status": "complete",
            "scan_window": scan_window,
            "collection_sha256": digest(collection_path),
            "collection_status": collection["status"],
            "decisions": decisions,
            "executions": executions,
            "blockers": [],
        }
        prepare_path = self.prepare_root / "prepare.json"
        prepare_path.write_text(json.dumps(prepare) + "\n")
        runner = {
            "schema_version": 1,
            "collection_sha256": digest(collection_path),
            "prepare_sha256": digest(prepare_path),
            "status": "complete",
            "results": results,
        }
        runner_path = self.runner_root / "runner/results.json"
        runner_path.parent.mkdir(exist_ok=True)
        runner_path.write_text(json.dumps(runner) + "\n")
        scan = {
            "schema_version": 1,
            "status": "complete",
            "collection_sha256": digest(collection_path),
            "collection_status": collection["status"],
            "prepare_sha256": digest(prepare_path),
            "runner_sha256": digest(runner_path),
            "environment": {"xpu_available": True},
            "candidates": candidates,
            "blockers": [],
        }
        scan_path = self.scan_root / "scan.json"
        scan_path.write_text(json.dumps(scan) + "\n")
        review = {
            "schema_version": 1,
            "status": "complete",
            "collection_sha256": digest(collection_path),
            "collection_status": collection["status"],
            "scan_sha256": digest(scan_path),
            "units": units,
            "blockers": [],
        }
        review_path = reviews / "review.json"
        review_path.write_text(json.dumps(review) + "\n")
        return {
            "collection": collection_path,
            "prepare": prepare_path,
            "runner": runner_path,
            "scan": scan_path,
            "review": review_path,
        }

    def make_partial(self, paths: dict[str, Path]) -> None:
        collection = json.loads(paths["collection"].read_text())
        collection["status"] = "partial"
        collection["sources"][0].update(
            {
                "status": "partial",
                "boundary_reached": False,
                "error": {"kind": "rate-limit", "message": "quota exhausted"},
            }
        )
        collection["blockers"] = ["issues-created:rate-limit"]
        paths["collection"].write_text(json.dumps(collection) + "\n")
        prepare = json.loads(paths["prepare"].read_text())
        prepare["collection_sha256"] = digest(paths["collection"])
        prepare["collection_status"] = "partial"
        paths["prepare"].write_text(json.dumps(prepare) + "\n")
        runner = json.loads(paths["runner"].read_text())
        runner["collection_sha256"] = digest(paths["collection"])
        runner["prepare_sha256"] = digest(paths["prepare"])
        paths["runner"].write_text(json.dumps(runner) + "\n")
        scan = json.loads(paths["scan"].read_text())
        scan["collection_sha256"] = digest(paths["collection"])
        scan["collection_status"] = "partial"
        scan["prepare_sha256"] = digest(paths["prepare"])
        scan["runner_sha256"] = digest(paths["runner"])
        paths["scan"].write_text(json.dumps(scan) + "\n")
        review = json.loads(paths["review"].read_text())
        review["collection_sha256"] = digest(paths["collection"])
        review["collection_status"] = "partial"
        review["scan_sha256"] = digest(paths["scan"])
        paths["review"].write_text(json.dumps(review) + "\n")


class AlignmentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.artifacts = AlignmentArtifacts(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def decision(self, *, mode: str = "schedule", producers_clean: bool = True) -> dict:
        return build_decision(
            self.artifacts.collection_root,
            self.artifacts.prepare_root,
            self.artifacts.runner_root,
            self.artifacts.scan_root,
            self.artifacts.review_root,
            mode=mode,
            producers_clean=producers_clean,
            run_id="42",
            scan_date="2026-08-20",
        )

    def test_complete_artifact_chain_files_one_reviewed_candidate(self) -> None:
        self.artifacts.write()

        decision = self.decision()

        self.assertEqual(decision["decision"], "file-one")
        self.assertEqual(decision["payloads"][0]["unit_id"], "issue-123")

    def test_partial_collection_returns_diagnostic_payloads(self) -> None:
        paths = self.artifacts.write()
        self.artifacts.make_partial(paths)

        decision = self.decision()

        self.assertEqual(decision["decision"], "diagnostic")
        self.assertEqual(decision["payloads"][0]["unit_id"], "issue-123")
        self.assertEqual(decision["collection_status"], "partial")
        self.assertIn("issues-created:rate-limit", decision["blockers"])

    def test_partial_dry_run_returns_a_diagnostic_decision(self) -> None:
        paths = self.artifacts.write()
        self.artifacts.make_partial(paths)

        decision = self.decision(mode="dry-run")

        self.assertEqual(decision["decision"], "dry-run-diagnostic")
        self.assertEqual(decision["would_decision"], "diagnostic")

    def test_inventory_count_mismatch_blocks_publishing(self) -> None:
        paths = self.artifacts.write()
        collection = json.loads(paths["collection"].read_text())
        collection["unique_count"] = 99
        paths["collection"].write_text(json.dumps(collection) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertTrue(any("inventory counts" in blocker for blocker in decision["blockers"]))

    def test_inventory_event_outside_scan_window_blocks_publishing(self) -> None:
        paths = self.artifacts.write()
        collection = json.loads(paths["collection"].read_text())
        collection["inventory"][0]["events"] = [
            {"type": "created", "at": "2026-08-19T23:59:59Z"}
        ]
        paths["collection"].write_text(json.dumps(collection) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertTrue(
            any(
                "outside the scan window" in blocker
                for blocker in decision["blockers"]
            )
        )

    def test_malformed_collection_timestamp_becomes_a_blocker(self) -> None:
        paths = self.artifacts.write()
        collection = json.loads(paths["collection"].read_text())
        collection["inventory"][0]["events"][0]["at"] = "not-a-timestamp"
        paths["collection"].write_text(json.dumps(collection) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertEqual(decision["payloads"], [])
        self.assertTrue(
            any("collection-invalid" in blocker for blocker in decision["blockers"])
        )

    def test_tampered_runner_log_blocks_publishing(self) -> None:
        self.artifacts.write()
        (self.artifacts.runner_root / "runner/logs/issue-123.log").write_text("tampered\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertIn("runner-log-digest-mismatch:issue-123", decision["blockers"])

    def test_two_reviewed_candidates_require_human_triage(self) -> None:
        self.artifacts.write(["issue-123", "issue-456"])

        decision = self.decision()

        self.assertEqual(decision["decision"], "triage")
        self.assertEqual(decision["actionable_units"], ["issue-123", "issue-456"])

    def test_dry_run_never_returns_a_filing_decision(self) -> None:
        self.artifacts.write()

        decision = self.decision(mode="dry-run")

        self.assertEqual(decision["decision"], "dry-run")
        self.assertEqual(decision["would_decision"], "file-one")
        self.assertEqual(decision["actionable_units"], ["issue-123"])

    def test_quiet_day_is_still_publishable_as_a_summary(self) -> None:
        self.artifacts.write([])

        scheduled = self.decision()
        dry_run = self.decision(mode="dry-run")

        self.assertEqual(scheduled["decision"], "none")
        self.assertEqual(dry_run["decision"], "dry-run")
        self.assertEqual(dry_run["would_decision"], "none")

    def test_review_must_cover_every_actionable_candidate(self) -> None:
        paths = self.artifacts.write(["issue-123", "issue-456"])
        review = json.loads(paths["review"].read_text())
        review["units"] = review["units"][:1]
        paths["review"].write_text(json.dumps(review) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertIn("review-coverage-mismatch", decision["blockers"])

    def test_failed_producer_blocks_all_candidate_publishing(self) -> None:
        self.artifacts.write()

        decision = self.decision(producers_clean=False)

        self.assertEqual(decision["decision"], "blocked")
        self.assertEqual(decision["payloads"], [])
        self.assertIn("producer-job-failed", decision["blockers"])

    def test_timed_out_reproducer_makes_the_scan_fail_closed(self) -> None:
        paths = self.artifacts.write()
        runner = json.loads(paths["runner"].read_text())
        runner["results"][0].update(
            {"returncode": None, "timed_out": True, "error": None}
        )
        paths["runner"].write_text(json.dumps(runner) + "\n")
        scan = json.loads(paths["scan"].read_text())
        scan["status"] = "incomplete"
        scan["runner_sha256"] = digest(paths["runner"])
        scan["candidates"][0].update(
            {"local_result": "blocked-script-error", "target_path_verified": False}
        )
        scan["blockers"] = ["issue-123 timed out"]
        paths["scan"].write_text(json.dumps(scan) + "\n")
        review = json.loads(paths["review"].read_text())
        review["scan_sha256"] = digest(paths["scan"])
        review["units"] = []
        paths["review"].write_text(json.dumps(review) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertEqual(decision["payloads"], [])
        self.assertIn("scan-not-complete:incomplete", decision["blockers"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
