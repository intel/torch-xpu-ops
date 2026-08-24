#!/usr/bin/env python3

import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from xpu_alignment_collect import collect
from xpu_alignment_runner import PlanError, _identity, load_prepare, run_plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prepare(root: Path, scripts: list[tuple[str, str]]) -> Path:
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
                        "title": unit_id,
                        "url": (
                            "https://github.com/pytorch/pytorch/issues/"
                            f"{unit_id.removeprefix('issue-')}"
                        ),
                        "event_at": "2026-08-20T01:00:00Z",
                    }
                    for unit_id, _ in scripts
                ]
            return {
                "nodes": nodes,
                "page_info": {"has_next_page": False, "end_cursor": None},
                "rate": {"remaining": 900, "reset_at": "2026-08-21T03:00:00Z"},
                "raw": {"nodes": nodes},
            }

    collection = collect("pytorch/pytorch", scan_window, root, GitHub())
    collection_path = root / "collection/collection.json"
    (root / "scripts").mkdir(parents=True)
    decisions = []
    executions = []
    for unit_id, source in scripts:
        script = root / f"scripts/repro_{unit_id}.py"
        script.write_text(source, encoding="utf-8")
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
                "script": str(script.relative_to(root)),
                "script_sha256": sha256(script),
                "timeout_seconds": 10,
                "oracle": "prints expected output",
                "target_path": "test target",
            }
        )
    prepare = root / "prepare.json"
    prepare.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "scan_window": scan_window,
                "collection_sha256": sha256(collection_path),
                "collection_status": collection["status"],
                "decisions": decisions,
                "executions": executions,
                "blockers": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return prepare


class RunnerTests(unittest.TestCase):
    def test_refuses_to_execute_reproducers_as_root(self) -> None:
        with self.assertRaisesRegex(PlanError, "must not run as root"):
            _identity("root")

    def test_executes_frozen_script_and_records_raw_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(root, [("issue-123", "print('repro ran')\n")])

            entries = load_prepare(root, prepare)
            result = run_plan(root, Path(sys.executable), prepare, entries)

            row = result["results"][0]
            self.assertEqual(row["returncode"], 0)
            self.assertFalse(row["timed_out"])
            self.assertEqual(row["script_sha256"], sha256(root / "scripts/repro_issue-123.py"))
            self.assertIn("repro ran", (root / row["log"]).read_text())

    def test_reproducer_cannot_read_agent_or_github_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(
                root,
                [
                    (
                        "issue-123",
                        "import os\n"
                        "print(os.getenv('GH_TOKEN'))\n"
                        "print(os.getenv('AWS_BEARER_TOKEN_BEDROCK'))\n"
                        "print(os.getenv('ANTHROPIC_API_KEY'))\n",
                    )
                ],
            )
            entries = load_prepare(root, prepare)

            with mock.patch.dict(
                os.environ,
                {
                    "GH_TOKEN": "github-secret",
                    "AWS_BEARER_TOKEN_BEDROCK": "cloud-secret",
                    "ANTHROPIC_API_KEY": "model-secret",
                },
            ):
                result = run_plan(root, Path(sys.executable), prepare, entries)

            log = (root / result["results"][0]["log"]).read_text()
            self.assertIn("None\nNone\nNone", log)
            self.assertNotIn("secret", log)

    def test_changed_script_is_not_executed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(root, [("issue-123", "print('approved')\n")])
            entries = load_prepare(root, prepare)
            (root / "scripts/repro_issue-123.py").write_text(
                "print('tampered')\n", encoding="utf-8"
            )

            result = run_plan(root, Path(sys.executable), prepare, entries)

            row = result["results"][0]
            self.assertIn("changed", row["error"])
            self.assertIsNone(row["returncode"])
            self.assertNotIn("tampered", (root / row["log"]).read_text())

    def test_timeout_does_not_prevent_later_reproducer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(
                root,
                [
                    ("issue-123", "import time\ntime.sleep(5)\n"),
                    ("issue-456", "print('second ran')\n"),
                ],
            )
            payload = json.loads(prepare.read_text())
            payload["executions"][0]["timeout_seconds"] = 1
            prepare.write_text(json.dumps(payload) + "\n")

            result = run_plan(
                root,
                Path(sys.executable),
                prepare,
                load_prepare(root, prepare),
            )

            self.assertTrue(result["results"][0]["timed_out"])
            self.assertEqual(result["results"][1]["returncode"], 0)
            self.assertIn("second ran", (root / result["results"][1]["log"]).read_text())

    def test_reproducer_cannot_leave_a_background_process_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            marker = root / "escaped-child"
            ready = root / "escaped-child-ready"
            child = (
                f"import os,pathlib,time; os.setsid(); pathlib.Path({str(ready)!r}).touch(); "
                "time.sleep(0.5); "
                f"pathlib.Path({str(marker)!r}).write_text('escaped')"
            )
            source = (
                "import pathlib,subprocess,sys,time\n"
                f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                f"ready = pathlib.Path({str(ready)!r})\n"
                "while not ready.exists(): time.sleep(0.01)\n"
            )
            prepare = write_prepare(root, [("issue-123", source)])

            run_plan(root, Path(sys.executable), prepare, load_prepare(root, prepare))
            time.sleep(0.8)

            self.assertFalse(marker.exists())

    def test_rejects_missing_execution_for_validated_inventory_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(root, [("issue-123", "print('ok')\n")])
            payload = json.loads(prepare.read_text())
            payload["executions"] = []
            prepare.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(PlanError, "coverage"):
                load_prepare(root, prepare)

    def test_rejects_incomplete_preparation_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(root, [("issue-123", "print('must not run')\n")])
            payload = json.loads(prepare.read_text())
            payload["status"] = "incomplete"
            payload["blockers"] = ["collection failed"]
            prepare.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(PlanError, "complete"):
                load_prepare(root, prepare)

    def test_accepts_partial_collection_for_diagnostic_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(root, [("issue-123", "print('diagnostic')\n")])
            collection_path = root / "collection/collection.json"
            collection = json.loads(collection_path.read_text())
            collection["status"] = "partial"
            collection["sources"][0].update(
                {
                    "status": "partial",
                    "boundary_reached": False,
                    "error": {"kind": "rate-limit", "message": "quota exhausted"},
                }
            )
            collection["blockers"] = ["issues-created:rate-limit"]
            collection_path.write_text(json.dumps(collection) + "\n")
            payload = json.loads(prepare.read_text())
            payload["collection_sha256"] = sha256(collection_path)
            payload["collection_status"] = "partial"
            prepare.write_text(json.dumps(payload) + "\n")

            entries = load_prepare(root, prepare)

            self.assertEqual([entry["id"] for entry in entries], ["issue-123"])

    def test_rejects_inventory_outside_scan_window_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(root, [("issue-123", "print('must not run')\n")])
            collection_path = root / "collection/collection.json"
            collection = json.loads(collection_path.read_text())
            collection["inventory"][0]["events"][0]["at"] = "2026-08-19T23:59:59Z"
            collection_path.write_text(json.dumps(collection) + "\n")

            with self.assertRaisesRegex(PlanError, "scan window"):
                load_prepare(root, prepare)


if __name__ == "__main__":
    unittest.main(verbosity=2)
