#!/usr/bin/env python3

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xpu_alignment_runner import PlanError, _identity, load_prepare, run_plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prepare(root: Path, scripts: list[tuple[str, str]]) -> Path:
    (root / "scripts").mkdir(parents=True)
    inventory = []
    executions = []
    for unit_id, source in scripts:
        script = root / f"scripts/repro_{unit_id}.py"
        script.write_text(source, encoding="utf-8")
        inventory.append(
            {
                "id": unit_id,
                "kind": "issue",
                "title": unit_id,
                "url": f"https://github.com/pytorch/pytorch/issues/{unit_id}",
                "events": [{"type": "created", "at": "2026-08-20T01:00:00Z"}],
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
                "scan_window": {
                    "start": "2026-08-20T00:00:00Z",
                    "end": "2026-08-21T00:00:00Z",
                },
                "collection": {"queries": []},
                "inventory": inventory,
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

    def test_rejects_missing_execution_for_validated_inventory_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prepare = write_prepare(root, [("issue-123", "print('ok')\n")])
            payload = json.loads(prepare.read_text())
            payload["executions"] = []
            prepare.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(PlanError, "coverage"):
                load_prepare(root, prepare)


if __name__ == "__main__":
    unittest.main(verbosity=2)
