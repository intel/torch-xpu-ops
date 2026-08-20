#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from xpu_alignment_gate import build_decision

SCAN = "2026-08-18"


def write_run(
    root: Path,
    *,
    ledger: list[dict[str, str]] | None = None,
    units: list[dict[str, str]] | None = None,
    payloads: list[str] | None = None,
    conclusions: bool = True,
    manifest: object | None = None,
    scope: str = SCAN,
) -> Path:
    """Lay out one restored run directory following the official skill layout."""
    run = root / scope
    artifacts = run / "artifacts"
    reports = run / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    if ledger is None:
        ledger = [done("candidate-1")]
    (artifacts / "candidate_ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger), encoding="utf-8"
    )

    if conclusions:
        (reports / "review_conclusions.md").write_text("# conclusions\n", encoding="utf-8")

    if manifest is None:
        manifest = {
            "units": units
            if units is not None
            else [{"id": "candidate-1", "verdict": "needs-xpu-fix"}]
        }
    (reports / "reviewer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    if payloads is None:
        payloads = [
            unit["id"]
            for unit in (units or [{"id": "candidate-1", "verdict": "needs-xpu-fix"}])
            if unit.get("verdict") == "needs-xpu-fix"
        ]
    for unit_id in payloads:
        write_payload(reports, unit_id)
    return run


def write_payload(reports: Path, unit_id: str, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "title": f"[xpu-alignment] {unit_id} diverges from CPU reference",
        "body": "## Summary\nDetails.\n",
        "labels": ["ai_generated"],
    }
    payload.update(overrides)
    path = reports / f"final_issue_{unit_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path

def done(unit_id: str) -> dict[str, str]:
    return {
        "id": unit_id,
        "title_status": "pass",
        "deep_status": "pass",
        "local_status": "done",
    }


def pending(unit_id: str) -> dict[str, str]:
    return {
        "id": unit_id,
        "title_status": "pass",
        "deep_status": "pending",
        "local_status": "pending",
    }


def rejected(unit_id: str) -> dict[str, str]:
    return {
        "id": unit_id,
        "title_status": "pass",
        "deep_status": "reject",
        "local_status": "pending",
    }


class AlignmentGateTests(unittest.TestCase):
    def decide(self, root: Path, **kwargs: object) -> dict[str, object]:
        options: dict[str, object] = {"auto_file": True, "scan_date": SCAN}
        options.update(kwargs)
        return build_decision(root, **options)  # type: ignore[arg-type]

    # --- the four normal outcomes -------------------------------------------------

    def test_single_actionable_unit_files_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root)
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "file-one")
            self.assertEqual(decision["actionable_units"], ["candidate-1"])
            self.assertFalse(decision["needs_attention"])
            self.assertEqual(len(decision["payloads"]), 1)

    def test_no_actionable_unit_is_a_quiet_green_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, units=[{"id": "candidate-1", "verdict": "non-issue"}])
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "none")
            self.assertFalse(decision["needs_attention"])
            self.assertEqual(decision["payloads"], [])

    def test_two_actionable_units_go_to_triage_and_stay_green(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(
                root,
                ledger=[done("candidate-a"), done("candidate-b")],
                units=[
                    {"id": "candidate-a", "verdict": "needs-xpu-fix"},
                    {"id": "candidate-b", "verdict": "needs-xpu-fix"},
                ],
            )
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "triage")
            self.assertEqual(decision["actionable_units"], ["candidate-a", "candidate-b"])
            # A busy day is normal, not a fault.
            self.assertFalse(decision["needs_attention"])
            self.assertEqual(len(decision["payloads"]), 2)

    def test_auto_file_disabled_sends_a_single_unit_to_triage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root)
            decision = self.decide(root, auto_file=False)
            self.assertEqual(decision["decision"], "triage")
            self.assertFalse(decision["needs_attention"])

    # --- completeness comes from the ledger ---------------------------------------

    def test_pending_row_closes_the_automatic_path_and_turns_red(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, ledger=[done("candidate-1"), pending("candidate-2")])
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "triage")
            self.assertFalse(decision["scan_complete"])
            self.assertTrue(decision["needs_attention"])
            self.assertEqual(decision["pending_units"], ["candidate-2"])

    def test_deep_rejected_row_is_not_pending_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, ledger=[done("candidate-1"), rejected("candidate-9")])
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "file-one")
            self.assertTrue(decision["scan_complete"])
            self.assertEqual(decision["pending_units"], [])

    def test_title_rejected_row_is_not_pending_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = [
                done("candidate-1"),
                {
                    "id": "candidate-8",
                    "title_status": "reject",
                    "deep_status": "pending",
                    "local_status": "pending",
                },
            ]
            decision = self.decide(root) if write_run(root, ledger=ledger) else None
            assert decision is not None
            self.assertEqual(decision["decision"], "file-one")
            self.assertEqual(decision["pending_units"], [])

    def test_missing_ledger_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root)
            next(root.glob("**/candidate_ledger.jsonl")).unlink()
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("ledger-missing", decision["blockers"])

    def test_unparsable_ledger_line_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root)
            path = next(root.glob("**/candidate_ledger.jsonl"))
            path.write_text(path.read_text(encoding="utf-8") + "{oops\n", encoding="utf-8")
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertTrue(any(b.startswith("ledger-unparsable") for b in decision["blockers"]))

    # --- review coverage ----------------------------------------------------------

    def test_review_that_skipped_a_done_row_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(
                root,
                ledger=[done("candidate-1"), done("candidate-2")],
                units=[{"id": "candidate-1", "verdict": "needs-xpu-fix"}],
            )
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("coverage-gap:candidate-2", decision["blockers"])

    def test_verdict_for_a_unit_absent_from_the_ledger_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(
                root,
                ledger=[done("candidate-1")],
                units=[
                    {"id": "candidate-1", "verdict": "needs-xpu-fix"},
                    {"id": "invented-1", "verdict": "needs-xpu-fix"},
                ],
            )
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("unknown-unit:invented-1", decision["blockers"])

    def test_missing_manifest_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root)
            next(root.glob("**/reviewer_manifest.json")).unlink()
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("reviewer-manifest-missing", decision["blockers"])

    def test_manifest_without_review_conclusions_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, conclusions=False)
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertTrue(
                any(b.startswith("manifest-missing-conclusions") for b in decision["blockers"])
            )

    def test_unknown_verdict_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(
                root,
                units=[{"id": "candidate-1", "verdict": "looks-bad"}],
                payloads=["candidate-1"],
            )
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("manifest-invalid-unit:candidate-1", decision["blockers"])

    def test_conflicting_verdicts_across_manifests_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root)
            write_run(
                root,
                scope="restored-copy",
                units=[{"id": "candidate-1", "verdict": "non-issue"}],
                payloads=[],
            )
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("conflicting-verdict:candidate-1", decision["blockers"])

    def test_unit_id_with_a_path_separator_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(
                root,
                ledger=[done("candidate-1")],
                units=[
                    {"id": "candidate-1", "verdict": "needs-xpu-fix"},
                    {"id": "../../escape", "verdict": "needs-xpu-fix"},
                ],
                payloads=["candidate-1"],
            )
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("manifest-invalid-unit:../../escape", decision["blockers"])

    # --- pre-generated payloads ---------------------------------------------------

    def test_missing_payload_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, payloads=[])
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("payload-not-unique:candidate-1:0", decision["blockers"])

    def test_payload_with_a_foreign_title_prefix_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = write_run(root, payloads=[])
            write_payload(run / "reports", "candidate-1", title="Fix everything")
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("payload-invalid-title:candidate-1", decision["blockers"])

    def test_payload_with_extra_labels_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = write_run(root, payloads=[])
            write_payload(run / "reports", "candidate-1", labels=["ai_generated", "bug"])
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("payload-invalid-labels:candidate-1", decision["blockers"])

    def test_payload_claiming_another_unit_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = write_run(root, payloads=[])
            path = run / "reports" / "final_issue_candidate-1.json"
            path.write_text(
                json.dumps(
                    {
                        "unit_id": "candidate-2",
                        "title": "[xpu-alignment] something",
                        "body": "body",
                        "labels": ["ai_generated"],
                    }
                ),
                encoding="utf-8",
            )
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("payload-unit-mismatch:candidate-1", decision["blockers"])

    def test_second_payload_for_the_same_unit_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root)
            duplicate = root / "restored-copy" / "reports"
            duplicate.mkdir(parents=True)
            write_payload(duplicate, "candidate-1")
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertIn("payload-not-unique:candidate-1:2", decision["blockers"])

    def test_a_blocked_run_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root, payloads=[])
            decision = self.decide(root)
            self.assertEqual(decision["decision"], "blocked")
            self.assertEqual(decision["payloads"], [])
            self.assertTrue(decision["needs_attention"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
