#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from xpu_alignment_gate import build_decision


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AlignmentArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.prepare_root = root / "prepare"
        self.runner_root = root / "runner"
        self.scan_root = root / "scan"
        self.review_root = root / "review"
        for path in (self.prepare_root, self.runner_root, self.scan_root, self.review_root):
            path.mkdir(parents=True)

    def write(self, unit_ids: list[str] | None = None) -> dict[str, Path]:
        unit_ids = ["issue-123"] if unit_ids is None else unit_ids
        scripts = self.prepare_root / "scripts"
        logs = self.runner_root / "runner/logs"
        reviews = self.review_root / "review"
        scripts.mkdir()
        logs.mkdir(parents=True)
        reviews.mkdir()
        inventory = []
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
            inventory.append(
                {
                    "id": unit_id,
                    "kind": "issue",
                    "title": f"Candidate {unit_id}",
                    "url": f"https://github.com/pytorch/pytorch/issues/{number}",
                    "events": [{"type": "created", "at": "2026-08-20T01:00:00Z"}],
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
        queries = []
        for source in (
            "issues-created",
            "prs-created",
            "prs-merged",
            "default-branch-commits",
        ):
            count = len(unit_ids) if source == "issues-created" else 0
            queries.append(
                {
                    "source": source,
                    "request": f"query {source}",
                    "pages": 1,
                    "count": count,
                    "truncated": False,
                    "errors": [],
                }
            )
        prepare = {
            "schema_version": 1,
            "status": "complete",
            "scan_window": {
                "start": "2026-08-20T00:00:00Z",
                "end": "2026-08-21T00:00:00Z",
            },
            "collection": {
                "queries": queries,
                "observed_count": sum(query["count"] for query in queries),
                "unique_count": len(inventory),
            },
            "inventory": inventory,
            "executions": executions,
            "blockers": [],
        }
        prepare_path = self.prepare_root / "prepare.json"
        prepare_path.write_text(json.dumps(prepare) + "\n")
        runner = {
            "schema_version": 1,
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
            "scan_sha256": digest(scan_path),
            "units": units,
            "blockers": [],
        }
        review_path = reviews / "review.json"
        review_path.write_text(json.dumps(review) + "\n")
        return {
            "prepare": prepare_path,
            "runner": runner_path,
            "scan": scan_path,
            "review": review_path,
        }


class AlignmentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.artifacts = AlignmentArtifacts(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def decision(self, *, mode: str = "schedule", producers_clean: bool = True) -> dict:
        return build_decision(
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

    def test_truncated_collection_blocks_all_candidate_publishing(self) -> None:
        paths = self.artifacts.write()
        prepare = json.loads(paths["prepare"].read_text())
        prepare["collection"]["queries"][0]["truncated"] = True
        paths["prepare"].write_text(json.dumps(prepare) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertEqual(decision["payloads"], [])
        self.assertIn("collection-query-truncated:0", decision["blockers"])

    def test_inventory_count_mismatch_blocks_publishing(self) -> None:
        paths = self.artifacts.write()
        prepare = json.loads(paths["prepare"].read_text())
        prepare["collection"]["unique_count"] = 99
        paths["prepare"].write_text(json.dumps(prepare) + "\n")

        decision = self.decision()

        self.assertEqual(decision["decision"], "blocked")
        self.assertIn("collection-unique-count-mismatch", decision["blockers"])

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
