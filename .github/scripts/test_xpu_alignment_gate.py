#!/usr/bin/env python3
import hashlib
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from xpu_alignment_gate import ACTIONABLE_RESULTS, build_decision


SCAN = "2026-08-18"


def candidate(
    unit_id: str,
    *,
    triage: str = "validate",
    validation: str = "done",
    result: str | None = "confirmed",
    category: str | None = None,
    duplicate_of: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": unit_id,
        "kind": "issue",
        "title": f"Candidate {unit_id}",
        "url": f"https://github.com/pytorch/pytorch/issues/{unit_id}",
        "events": [{"type": "created", "at": "2026-08-18T01:00:00Z"}],
        "triage_status": triage,
        "triage_reason": "not relevant" if triage == "reject" else "worth validating",
        "rejection_category": category,
        "duplicate_of": duplicate_of,
        "validation_status": validation,
        "local_result": result,
        "repro_path": f"scripts/repro_{unit_id}.py" if triage == "validate" else None,
        "log_path": f"artifacts/output_{unit_id}.log" if triage == "validate" else None,
    }


def rejected(
    unit_id: str, category: str = "nonfunctional", duplicate_of: str | None = None
) -> dict[str, object]:
    return candidate(
        unit_id,
        triage="reject",
        validation="not-needed",
        result=None,
        category=category,
        duplicate_of=duplicate_of,
    )


def pending(unit_id: str) -> dict[str, object]:
    return candidate(unit_id, validation="pending", result=None)


def scan_manifest(*, status: str = "complete", collection: str = "complete") -> dict:
    return {
        "schema_version": 1,
        "mode": "automation",
        "phase": "final",
        "status": status,
        "scan_window": {
            "start": "2026-08-18T00:00:00Z",
            "end": "2026-08-19T00:00:00Z",
            "timezone": "UTC",
        },
        "environment": {
            "python": "/usr/bin/python3",
            "torch": "2.9.0.dev",
            "xpu_available": True,
            "device": "Intel GPU",
        },
        "collection": {
            "status": collection,
            "sources": {
                "issues": {
                    "event_types": ["created"],
                    "queries": [
                        {
                            "request": "issues created",
                            "pages": 1,
                            "count": 1,
                            "truncated": collection != "complete",
                        }
                    ],
                    "pages": 1,
                    "count": 1,
                    "truncated": collection != "complete",
                },
                "prs": {
                    "event_types": ["created", "merged"],
                    "queries": [
                        {"request": "prs created", "pages": 1, "count": 1, "truncated": False},
                        {"request": "prs merged", "pages": 1, "count": 0, "truncated": False},
                    ],
                    "pages": 2,
                    "count": 1,
                    "truncated": False,
                },
                "commits": {
                    "event_types": ["default-branch"],
                    "queries": [
                        {
                            "request": "default branch commits",
                            "pages": 1,
                            "count": 1,
                            "truncated": False,
                        }
                    ],
                    "pages": 1,
                    "count": 1,
                    "truncated": False,
                },
            },
            "errors": [] if collection == "complete" else ["issues-truncated"],
        },
        "raw_candidates": "artifacts/raw_candidates.json",
        "ledger": "artifacts/candidate_ledger.jsonl",
        "execution_plan": "artifacts/execution_plan.json",
        "execution_results": "artifacts/execution_results.json",
        "blockers": [],
    }


def samples_for(rows: list[dict[str, object]], count: int = 3) -> list[dict[str, str]]:
    categories: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["triage_status"] == "reject":
            categories[str(row["rejection_category"])].append(str(row["id"]))
        elif row["validation_status"] == "done" and row["local_result"] not in ACTIONABLE_RESULTS:
            categories[str(row["local_result"])].append(str(row["id"]))
    return [
        {"id": unit_id, "category": category, "outcome": "accepted"}
        for category, ids in sorted(categories.items())
        for unit_id in sorted(ids)[:count]
    ]


def review_unit(unit_id: str, verdict: str = "needs-xpu-fix", **overrides: object) -> dict:
    entry: dict[str, object] = {
        "id": unit_id,
        "verdict": verdict,
        "implementation_repository": "intel/torch-xpu-ops",
        "canonical_tracker": None,
        "payload": f"review/final_issue_{unit_id}.json" if verdict == "needs-xpu-fix" else None,
    }
    entry.update(overrides)
    return entry


def write_payload(run: Path, unit_id: str, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "title": f"[xpu-alignment] {unit_id} diverges from the reference",
        "body": "## Summary\nReviewed XPU evidence.\n",
        "labels": ["ai_generated"],
    }
    payload.update(overrides)
    path = run / "review" / f"final_issue_{unit_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_run(
    review_root: Path,
    scan_root: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    scan: dict | None = None,
    units: list[dict] | None = None,
    promoted: list[str] | None = None,
    review_overrides: dict | None = None,
) -> tuple[Path, Path]:
    rows = rows or [candidate("candidate-1")]
    scan = scan or scan_manifest()
    promoted = promoted or []
    source_names = {"issue": "issues", "pr": "prs", "commit": "commits"}
    for source in source_names.values():
        scan["collection"]["sources"][source]["count"] = 0
    for row in rows:
        source = source_names[str(row["kind"])]
        scan["collection"]["sources"][source]["count"] += 1
    sources = scan["collection"]["sources"]
    sources["issues"]["queries"][0]["count"] = sources["issues"]["count"]
    sources["prs"]["queries"][0]["count"] = sources["prs"]["count"]
    sources["commits"]["queries"][0]["count"] = sources["commits"]["count"]
    review_run = review_root / SCAN
    immutable_run = scan_root / SCAN
    for run in (review_run, immutable_run):
        (run / "artifacts").mkdir(parents=True, exist_ok=True)
        (run / "scripts").mkdir(parents=True, exist_ok=True)
        (run / "reports").mkdir(parents=True, exist_ok=True)
        (run / "scan_manifest.json").write_text(json.dumps(scan), encoding="utf-8")
        (run / "artifacts/candidate_ledger.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        raw_candidates = [
            {
                "id": row["id"],
                "kind": row["kind"],
                "title": row["title"],
                "url": row["url"],
                "events": row["events"],
            }
            for row in rows
        ]
        (run / "artifacts/raw_candidates.json").write_text(
            json.dumps(raw_candidates), encoding="utf-8"
        )
        (run / "reports/scan_report.md").write_text("# Scan report\n", encoding="utf-8")
        plan_entries: list[dict[str, object]] = []
        result_entries: list[dict[str, object]] = []
        for row in rows:
            if row["triage_status"] != "validate":
                continue
            unit_id = str(row["id"])
            script_path = run / str(row["repro_path"])
            script_path.write_text(f"print({unit_id!r})\n", encoding="utf-8")
            script_digest = hashlib.sha256(script_path.read_bytes()).hexdigest()
            plan_entries.append(
                {
                    "id": unit_id,
                    "path": row["repro_path"],
                    "log_path": row["log_path"],
                    "timeout_seconds": 120,
                    "sha256": script_digest,
                    "precheck_status": "approved",
                    "upstream_oracle": "test oracle",
                    "target_xpu_path": "test target",
                    "xpu_proof": "test proof",
                }
            )
            (run / str(row["log_path"])).write_text("target xpu proof\n", encoding="utf-8")
            result_entries.append(
                {
                    "id": unit_id,
                    "script_path": row["repro_path"],
                    "log_path": row["log_path"],
                    "sha256": script_digest,
                    "runner_status": "completed",
                    "timed_out": False,
                    "returncode": 0,
                    "signal": None,
                    "duration_seconds": 0.1,
                    "started_at": "2026-08-18T02:00:00Z",
                    "finished_at": "2026-08-18T02:00:01Z",
                    "error": None,
                }
            )
        (run / "artifacts/execution_plan.json").write_text(
            json.dumps({"schema_version": 1, "scripts": plan_entries}), encoding="utf-8"
        )
        (run / "artifacts/execution_results.json").write_text(
            json.dumps({"schema_version": 1, "results": result_entries}), encoding="utf-8"
        )

    mandatory = sorted(
        str(row["id"])
        for row in rows
        if row["validation_status"] == "done" and row["local_result"] in ACTIONABLE_RESULTS
    )
    if units is None:
        units = [review_unit(unit_id) for unit_id in mandatory]
    review = {
        "schema_version": 1,
        "review_status": "complete",
        "sample_policy": {"per_category": 3, "order": "id-lexical"},
        "mandatory_units": mandatory,
        "negative_samples": samples_for(rows),
        "promoted_units": promoted,
        "units": units,
        "blockers": [],
    }
    for sample in review["negative_samples"]:
        if sample["id"] in promoted:
            sample["outcome"] = "promoted"
    if review_overrides:
        review.update(review_overrides)
    review_dir = review_run / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "review_manifest.json").write_text(json.dumps(review), encoding="utf-8")
    (review_dir / "review_report.md").write_text("# Independent review\n", encoding="utf-8")
    for entry in units:
        if entry.get("payload"):
            write_payload(review_run, str(entry["id"]))
    return review_run, immutable_run


class AlignmentGateTests(unittest.TestCase):
    def decide(self, review_root: Path, scan_root: Path, **kwargs: object) -> dict[str, object]:
        options: dict[str, object] = {
            "auto_file": True,
            "scan_date": SCAN,
            "scan_root": scan_root,
        }
        options.update(kwargs)
        return build_decision(review_root, **options)  # type: ignore[arg-type]

    def test_one_publishable_unit_files_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = root / "reviewed"
            scan = root / "scan"
            write_run(review, scan)
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "file-one")
            self.assertEqual(decision["actionable_units"], ["candidate-1"])
            self.assertFalse(decision["needs_attention"])

    def test_no_actionable_unit_is_a_quiet_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan, rows=[candidate("candidate-1", result="not-reproduced")])
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "none")
            self.assertFalse(decision["needs_attention"])

    def test_two_publishable_units_go_to_triage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan, rows=[candidate("candidate-a"), candidate("candidate-b")])
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "triage")
            self.assertEqual(decision["actionable_units"], ["candidate-a", "candidate-b"])
            self.assertFalse(decision["needs_attention"])

    def test_auto_file_false_routes_one_unit_to_triage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan)
            self.assertEqual(self.decide(review, scan, auto_file=False)["decision"], "triage")

    def test_incomplete_collection_disables_auto_file_but_keeps_triage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(
                review,
                scan,
                scan=scan_manifest(status="incomplete", collection="incomplete"),
            )
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "triage")
            self.assertFalse(decision["collection_complete"])
            self.assertTrue(decision["needs_attention"])

    def test_pending_validation_disables_auto_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan, rows=[candidate("candidate-1"), pending("candidate-2")])
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "triage")
            self.assertEqual(decision["pending_units"], ["candidate-2"])
            self.assertTrue(decision["needs_attention"])

    def test_blocked_validation_disables_auto_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(
                review,
                scan,
                rows=[candidate("candidate-1"), candidate("candidate-2", result="blocked-env")],
            )
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "triage")
            self.assertEqual(decision["unresolved_units"], ["candidate-2"])
            self.assertIn("unresolved-validation", decision["attention_reasons"])

    def test_prepared_scan_disables_auto_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            manifest = scan_manifest(status="incomplete")
            manifest["phase"] = "prepared"
            write_run(review, scan, scan=manifest)
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "triage")
            self.assertIn("scan-not-final", decision["attention_reasons"])

    def test_gate_date_must_match_manifest_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan)
            decision = self.decide(review, scan, scan_date="2026-08-17")
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("scan-window-does-not-match-gate-date", decision["blockers"])

    def test_missing_scan_manifest_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan)
            (scan / SCAN / "scan_manifest.json").unlink()
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("scan-manifest-count:0", decision["blockers"])

    def test_malformed_ledger_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan)
            (scan / SCAN / "artifacts/candidate_ledger.jsonl").write_text("not json\n")
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertTrue(any("ledger-unparsable" in item for item in decision["blockers"]))

    def test_raw_candidate_set_must_equal_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan)
            (scan / SCAN / "artifacts/raw_candidates.json").write_text("[]\n", encoding="utf-8")
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("raw-candidate-ledger-id-mismatch", decision["blockers"])

    def test_query_evidence_must_match_source_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            manifest = scan_manifest()
            manifest["collection"]["sources"]["issues"]["queries"][0]["pages"] = 2
            write_run(review, scan, scan=manifest)
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("scan-source-page-total:issues", decision["blockers"])

    def test_missing_scan_report_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan)
            (scan / SCAN / "reports/scan_report.md").unlink()
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("scan-report-missing", decision["blockers"])

    def test_changed_repro_after_execution_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan)
            (scan / SCAN / "scripts/repro_candidate-1.py").write_text(
                "print('changed')\n", encoding="utf-8"
            )
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("execution-script-digest:candidate-1", decision["blockers"])

    def test_missing_mandatory_verdict_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan, units=[], review_overrides={"mandatory_units": []})
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("review-mandatory-set-mismatch", decision["blockers"])

    def test_review_outputs_must_match_the_scan_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            review_run, _ = write_run(review, scan)
            wrong_run = review / "wrong-date"
            wrong_run.mkdir(parents=True)
            (review_run / "review").rename(wrong_run / "review")
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("review-run-mismatch", decision["blockers"])

    def test_unknown_review_unit_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(
                review,
                scan,
                units=[review_unit("candidate-1"), review_unit("invented", "non-issue")],
            )
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertTrue(any("review-invalid-unit:invented" in item for item in decision["blockers"]))

    def test_negative_sample_mismatch_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(
                review,
                scan,
                rows=[candidate("candidate-1", result="not-reproduced")],
                review_overrides={"negative_samples": []},
            )
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("review-negative-sample-mismatch", decision["blockers"])

    def test_promoted_negative_gets_a_formal_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(
                review,
                scan,
                rows=[candidate("candidate-1"), rejected("candidate-2")],
                promoted=["candidate-2"],
                units=[
                    review_unit("candidate-1"),
                    review_unit("candidate-2", "verification-gap"),
                ],
            )
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "triage")
            self.assertEqual(decision["promoted_units"], ["candidate-2"])
            self.assertIn("negative-sample-promotion", decision["attention_reasons"])

    def test_existing_canonical_tracker_needs_no_new_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            unit = review_unit(
                "candidate-1",
                canonical_tracker="https://github.com/intel/torch-xpu-ops/issues/42",
                payload=None,
            )
            write_run(review, scan, units=[unit])
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "none")
            self.assertEqual(decision["needs_xpu_fix_units"], ["candidate-1"])

    def test_verification_gap_disables_unattended_filing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            rows = [candidate("candidate-1"), candidate("candidate-2")]
            units = [
                review_unit("candidate-1"),
                review_unit("candidate-2", "verification-gap"),
            ]
            write_run(review, scan, rows=rows, units=units)
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "triage")
            self.assertEqual(decision["review_unresolved_units"], ["candidate-2"])
            self.assertIn("review-verification-gap", decision["attention_reasons"])

    def test_blocked_review_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(
                review,
                scan,
                review_overrides={"review_status": "blocked", "blockers": ["quota"]},
            )
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertEqual(decision["payloads"], [])

    def test_invalid_payload_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            review_run, _ = write_run(review, scan)
            write_payload(review_run, "candidate-1", labels=["unexpected"])
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("payload-invalid-labels:candidate-1", decision["blockers"])

    def test_payload_authored_in_scan_artifact_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            _, scan_run = write_run(review, scan)
            write_payload(scan_run, "candidate-1")
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("payload-authored-before-review:candidate-1", decision["blockers"])

    def test_undeclared_payload_file_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            review_run, _ = write_run(review, scan)
            write_payload(review_run, "extra")
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("review-payload-file-set-mismatch", decision["blockers"])

    def test_duplicate_chain_is_sampled_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            rows = [candidate("primary"), rejected("copy", "duplicate-chain", "primary")]
            write_run(review, scan, rows=rows)
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "file-one")
            self.assertEqual(decision["blockers"], [])

    def test_duplicate_chain_target_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            write_run(review, scan, rows=[rejected("copy", "duplicate-chain", "missing")])
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("ledger-duplicate-target-missing:copy", decision["blockers"])

    def test_old_schema_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review, scan = root / "reviewed", root / "scan"
            manifest = scan_manifest()
            manifest["schema_version"] = 0
            write_run(review, scan, scan=manifest)
            decision = self.decide(review, scan)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("scan-manifest-invalid-version", decision["blockers"])


if __name__ == "__main__":
    unittest.main()
