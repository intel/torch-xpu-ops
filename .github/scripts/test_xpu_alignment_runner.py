#!/usr/bin/env python3
# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xpu_alignment_runner import PlanError, _identity, load_plan, run_plan


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_plan(root: Path, script_text: str, **overrides: object) -> Path:
    scripts = root / "scripts"
    artifacts = root / "artifacts"
    scripts.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    script = scripts / "repro_candidate-1.py"
    script.write_text(script_text, encoding="utf-8")
    entry: dict[str, object] = {
        "id": "candidate-1",
        "path": "scripts/repro_candidate-1.py",
        "log_path": "artifacts/output_candidate-1.log",
        "timeout_seconds": 10,
        "sha256": digest(script),
        "precheck_status": "approved",
        "upstream_oracle": "prints ok",
        "target_xpu_path": "test-only path",
        "xpu_proof": "test-only proof",
    }
    entry.update(overrides)
    plan = artifacts / "execution_plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "scripts": [entry]}), encoding="utf-8")
    return plan


class RunnerTests(unittest.TestCase):
    def test_refuses_root_as_the_reproducer_identity(self) -> None:
        with self.assertRaisesRegex(PlanError, "must not run as root"):
            _identity("root")

    def test_executes_approved_script_and_records_raw_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write_plan(root, "print('repro ran')\n")
            entries = load_plan(root, plan)
            result = run_plan(root, Path(sys.executable), entries)
            self.assertEqual(result["results"][0]["runner_status"], "completed")
            self.assertEqual(result["results"][0]["returncode"], 0)
            self.assertIn("repro ran", (root / "artifacts/output_candidate-1.log").read_text())

    def test_strips_credentials_from_reproducer_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write_plan(
                root,
                "import os\n"
                "print(os.getenv('GH_TOKEN'))\n"
                "print(os.getenv('AWS_TEST_SECRET'))\n"
                "print(os.getenv('ANTHROPIC_API_KEY'))\n",
            )
            entries = load_plan(root, plan)
            secrets = {
                "GH_TOKEN": "secret",
                "AWS_TEST_SECRET": "cloud",
                "ANTHROPIC_API_KEY": "model-key",
            }
            with mock.patch.dict(os.environ, secrets):
                run_plan(root, Path(sys.executable), entries)
            log = (root / "artifacts/output_candidate-1.log").read_text()
            self.assertNotIn("secret", log)
            self.assertNotIn("cloud", log)
            self.assertNotIn("model-key", log)
            self.assertIn("None\nNone\nNone", log)

    def test_rejects_changed_script_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write_plan(root, "print('before')\n")
            (root / "scripts/repro_candidate-1.py").write_text("print('after')\n")
            with self.assertRaisesRegex(PlanError, "digest"):
                load_plan(root, plan)

    def test_rechecks_digest_immediately_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write_plan(root, "print('approved bytes')\n")
            entries = load_plan(root, plan)
            (root / "scripts/repro_candidate-1.py").write_text(
                "print('changed bytes must not run')\n", encoding="utf-8"
            )
            result = run_plan(root, Path(sys.executable), entries)
            row = result["results"][0]
            self.assertEqual(row["runner_status"], "integrity-error")
            log = (root / "artifacts/output_candidate-1.log").read_text()
            self.assertNotIn("changed bytes must not run", log)

    def test_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write_plan(root, "print('ok')\n", path="../outside.py")
            with self.assertRaisesRegex(PlanError, "unsafe relative path"):
                load_plan(root, plan)

    def test_rejects_unapproved_script(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write_plan(root, "print('ok')\n", precheck_status="rework")
            with self.assertRaisesRegex(PlanError, "approved"):
                load_plan(root, plan)

    def test_timeout_is_a_runner_observation_not_a_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write_plan(root, "import time\ntime.sleep(5)\n", timeout_seconds=1)
            entries = load_plan(root, plan)
            result = run_plan(root, Path(sys.executable), entries)
            row = result["results"][0]
            self.assertEqual(row["runner_status"], "timeout")
            self.assertTrue(row["timed_out"])
            self.assertNotIn("local_result", row)


if __name__ == "__main__":
    unittest.main()
