# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ut_results  # noqa: E402


class P0ResultsTest(unittest.TestCase):
    def test_p0_check_and_summary(self):
        for job in ("xpu_p0", "xpu_p0_distributed"):
            for failure in (False, True):
                with self.subTest(job=job, failure=failure), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    artifact = root / "data" / f"UT-Data-linux-{job}-1"
                    artifact.mkdir(parents=True)
                    (artifact / "ut_meta.env").write_text(f"UT_NAME={job}\nUT_RUNNER=bmg-test\n")
                    outcome = '<error message="collection failure">ImportError: missing</error>' if failure else ""
                    (artifact / "xpu_p0_with_test.xml").write_text(
                        '<testsuites><testsuite tests="2" failures="0" errors="'
                        f'{int(failure)}" skipped="0"><testcase classname="mod.Test" name="test_pass"/>'
                        f'<testcase classname="mod.Test" name="test_case">{outcome}</testcase>'
                        '</testsuite></testsuites>'
                    )
                    checks = root / "checks"
                    with mock.patch.object(ut_results.KnownIssues, "_fetch", return_value=""):
                        rc = ut_results.check_run(root / "data", root / "snapshot", root / "work",
                                                  checks, "release_wheel")
                    verdict = json.loads((checks / job / "verdict.json").read_text())["categories"][0]
                    self.assertEqual(rc, int(failure))
                    self.assertEqual(verdict["category"], job)
                    self.assertEqual(verdict["total"], 2)
                    self.assertEqual(verdict["new"], int(failure))
                    self.assertEqual(verdict["passed"], not failure)

                    self.assertEqual(ut_results.render_report(checks, root / "report"), 0)
                    summary = (root / "report" / "ut_report.md").read_text()
                    self.assertIn(f"| linux | {job} | 2 | 0 | {int(failure)} |", summary)
                    if failure:
                        self.assertIn("test_case", summary)

    def test_unknown_suite_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(ut_results.Checker(Path(tmp)).run("unrecognized_suite").passed)

    def test_p0_without_reports_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "data" / "UT-Data-linux-xpu_p0-1"
            artifact.mkdir(parents=True)
            (artifact / "ut_meta.env").write_text("UT_NAME=xpu_p0\n")
            with mock.patch.object(ut_results.KnownIssues, "_fetch", return_value=""):
                rc = ut_results.check_run(root / "data", root / "snapshot", root / "work",
                                          root / "checks", "release_wheel")
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
