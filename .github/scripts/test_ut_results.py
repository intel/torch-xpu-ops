"""Unit tests for ut_results.py.

Covers what the CI depends on and cannot see until a nightly breaks: the shape
of the case logs, the known-failure comparison and the pass/fail verdict.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ut_results", SCRIPTS / "ut_results.py")
ut_results = importlib.util.module_from_spec(spec)
sys.modules["ut_results"] = ut_results
spec.loader.exec_module(ut_results)


def write_xml(path: Path, cases, tests=None, failures=0, errors=0, skipped=0):
    body = "".join(cases)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<testsuites><testsuite name="pytest" tests="{tests if tests is not None else len(cases)}"'
        f' failures="{failures}" errors="{errors}" skipped="{skipped}" time="1">'
        f"{body}</testsuite></testsuites>",
        encoding="utf-8",
    )


PASS = '<testcase classname="TestFoo" name="test_ok" time="0.1"/>'
SKIP = '<testcase classname="TestFoo" name="test_skip"><skipped message="nope"/></testcase>'
FAIL = ('<testcase classname="TestFoo" name="test_bad">'
        '<failure message="boom">RuntimeError: boom\n  at line 1</failure></testcase>')


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def test_report_writes_one_log_per_status(tmp_path):
    write_xml(tmp_path / "op_regression.xml", [PASS, SKIP, FAIL], failures=1, skipped=1)

    report = ut_results.build_report("op_regression", [tmp_path / "op_regression.xml"], tmp_path)

    assert (tmp_path / "passed_op_regression.log").read_text() == \
        "op_regression,TestFoo,test_ok\n"
    assert (tmp_path / "failures_op_regression.log").read_text() == \
        "op_regression,TestFoo,test_bad\n"
    # The skipped case exists only here, which is how a newly enabled case is
    # told apart from a brand-new one.
    assert (tmp_path / "all_cases_op_regression.log").read_text().splitlines() == [
        "op_regression,TestFoo,test_ok",
        "op_regression,TestFoo,test_skip",
        "op_regression,TestFoo,test_bad",
    ]
    assert report.totals()["op_regression"] == {
        "Test cases": 3, "Passed": 1, "Skipped": 1, "Failures": 1, "Errors": 0,
    }


def test_report_failure_csv_is_headerless_and_carries_the_message(tmp_path):
    write_xml(tmp_path / "op_ut_with_skip.xml", [FAIL], failures=1)

    ut_results.build_report("op_ut", [tmp_path / "op_ut_with_skip.xml"], tmp_path)

    rows = (tmp_path / "ut_failure_list.csv").read_text().splitlines()
    assert rows == ["| op_ut | TestFoo | test_bad | failed | RuntimeError: boom | XML |"]


def test_report_skips_the_failure_csv_for_skipped_ut(tmp_path):
    write_xml(tmp_path / "op_ut_with_skip.xml", [FAIL], failures=1)

    ut_results.build_report("skipped_ut", [tmp_path / "op_ut_with_skip.xml"], tmp_path)

    assert not (tmp_path / "ut_failure_list.csv").exists()


def test_report_category_totals_log_round_trips_through_the_checker(tmp_path):
    write_xml(tmp_path / "op_regression.xml", [PASS] * 353, tests=353)

    ut_results.build_report("op_regression", [tmp_path / "op_regression.xml"], tmp_path)
    checker = ut_results.Checker(work_dir=tmp_path, health_file=tmp_path / "health.jsonl")

    assert checker.check_case_counts("op_regression") is True
    record = json.loads((tmp_path / "health.jsonl").read_text())
    assert record == {"category": "op_regression", "expected": 353, "actual": 353,
                      "ratio": 1.0, "healthy": True, "reason": ""}


def test_report_survives_a_broken_file(tmp_path, capsys):
    (tmp_path / "op_ut.xml").write_text("<not xml", encoding="utf-8")
    write_xml(tmp_path / "op_regression.xml", [PASS])

    report = ut_results.build_report(
        "basic", [tmp_path / "op_ut.xml", tmp_path / "op_regression.xml"], tmp_path)

    assert "Error processing" in capsys.readouterr().err
    assert len(report.cases) == 1


# --------------------------------------------------------------------------- #
# report markdown
# --------------------------------------------------------------------------- #


def two_failures(tmp_path):
    write_xml(tmp_path / "op_regression.xml", [
        PASS,
        FAIL,
        '<testcase classname="TestFoo" name="test_known">'
        '<failure message="old">RuntimeError: old</failure></testcase>',
    ], failures=2)
    return ut_results.build_report(
        "op_regression", [tmp_path / "op_regression.xml"], tmp_path)


def test_failures_show_the_new_ones_and_fold_the_known_ones(tmp_path):
    report = two_failures(tmp_path)

    markdown = report.render({"op_regression,TestFoo,test_bad"})

    shown, _, folded = markdown.partition("<details>")
    assert "test_bad" in shown
    assert "test_known" not in shown
    assert "<summary>1 known failure</summary>" in folded
    assert "test_known" in folded


def test_failures_say_so_when_every_failure_is_known(tmp_path):
    report = two_failures(tmp_path)

    markdown = report.render(set())

    assert "✅ No new failures." in markdown.partition("<details>")[0]
    assert "<summary>2 known failures</summary>" in markdown


def test_failures_are_all_shown_without_a_comparison(tmp_path):
    report = two_failures(tmp_path)

    markdown = report.render()

    assert "<details>" not in markdown.split("### Results Summary")[0]
    assert "test_bad" in markdown
    assert "test_known" in markdown


def test_summary_shows_the_total_and_folds_the_rest(tmp_path):
    write_xml(tmp_path / "op_regression.xml", [PASS])
    write_xml(tmp_path / "op_extended.xml", [PASS, FAIL], failures=1)
    report = ut_results.build_report(
        "basic", [tmp_path / "op_extended.xml", tmp_path / "op_regression.xml"], tmp_path)

    summary = "\n".join(report.summary_md())
    shown, _, folded = summary.partition("<details>")

    assert shown.splitlines()[0] == \
        "| Category | Test cases | Passed | Skipped | Failures | Errors |"
    assert "| **Total** | 3 | 2 | 0 | 1 | 0 |" in shown
    assert "op_extended" not in shown
    # The category and per-report rows are still there, one fold away.
    assert "<summary>2 categories</summary>" in folded
    assert "| op_extended | 2 | 1 | 0 | 1 | 0 |" in folded
    assert "| op_regression | 1 | 1 | 0 | 0 | 0 |" in folded
    assert "<summary>2 reports</summary>" in folded
    assert "| op_extended | op_extended | 2 | 1 | 0 | 1 | 0 | XML |" in folded


@pytest.mark.parametrize(("name", "expected"), [
    ("op_regression", "op_regression"),
    ("op_ut_with_skip_test_xpu", "op_ut"),
    ("upstream_inductor_test_foo", "upstream_inductor"),
    ("xpu_profiling_rn50", "xpu_profiling"),
    ("something_else", "unknown"),
])
def test_determine_category(name, expected):
    assert ut_results.determine_category(name) == expected


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #


def seed(tmp_path, *, failures=(), passed=(), known=(), cases=353):
    ut_results.write_lines(tmp_path / "failures_op_regression.log", list(failures))
    ut_results.write_lines(tmp_path / "passed_op_regression.log", list(passed))
    ut_results.write_lines(tmp_path / "Known_issue.log", list(known))
    ut_results.write_lines(tmp_path / "category_op_regression.log", [
        "Category: op_regression", f"Test cases: {cases}", "Passed: 1",
        "Skipped: 0", "Failures: 0", "Errors: 0",
    ])
    return ut_results.Checker(work_dir=tmp_path)


def test_check_passes_when_every_failure_is_known(tmp_path):
    checker = seed(tmp_path,
                   failures=["op_regression,TestFoo,test_bad"],
                   passed=["op_regression,TestFoo,test_ok"],
                   known=["op_regression,TestFoo,test_bad"])

    result = checker.run("op_regression")

    assert result.passed
    assert result.new_failures == []
    assert (tmp_path / "failures_op_regression_filtered.log").read_text() == ""


def test_check_fails_on_a_new_failure(tmp_path):
    checker = seed(tmp_path,
                   failures=["op_regression,TestFoo,test_new"],
                   passed=["op_regression,TestFoo,test_ok"],
                   known=["op_regression,TestFoo,test_bad"])

    result = checker.run("op_regression")

    assert not result.passed
    assert result.new_failures == ["op_regression,TestFoo,test_new"]


def test_check_fails_when_nothing_passed(tmp_path):
    assert not seed(tmp_path, passed=[]).run("op_regression").passed


def test_check_fails_when_the_case_count_collapsed(tmp_path):
    checker = seed(tmp_path, passed=["op_regression,TestFoo,test_ok"], cases=100)

    result = checker.run("op_regression")

    assert not result.passed
    assert result.counts_normal is False


def test_check_tolerates_a_missing_category_log(tmp_path):
    checker = seed(tmp_path, passed=["op_regression,TestFoo,test_ok"])
    (tmp_path / "category_op_regression.log").unlink()

    result = checker.run("op_regression")

    assert result.counts_normal is None
    assert result.passed


def test_check_does_not_touch_github_outside_a_scheduled_source_build(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    checker = seed(tmp_path)
    called = []
    monkeypatch.setattr(ut_results, "gh", lambda *a, **k: called.append(a))

    checker.report_passing_known_issues(["op_regression,TestFoo,test_ok"])

    assert called == []


def test_check_unknown_category_is_not_a_failure(tmp_path, capsys):
    assert ut_results.Checker(work_dir=tmp_path).run("nope").passed
    assert "Unknown test suite" in capsys.readouterr().err


def test_distributed_check_reads_the_console_log(tmp_path):
    (tmp_path / "xpu_distributed_test.log").write_text(
        "PASSED test/foo.py::TestBar::test_ok\n"
        "FAILED test/foo.py::TestBar::test_bad\n"
        "PASSED test/foo.py::TestBar::test_ok\n",
        encoding="utf-8",
    )
    ut_results.write_lines(tmp_path / "Known_issue.log", [])
    checker = ut_results.Checker(work_dir=tmp_path)

    result = checker.run("xpu_distributed")

    assert not result.passed
    assert result.new_failures == ["test/foo.py::TestBar::test_bad"]
    assert (tmp_path / "xpu_distributed_passed.log").read_text() == \
        "test/foo.py::TestBar::test_ok\n"


def test_profiling_check_fails_on_a_failing_report(tmp_path):
    write_xml(tmp_path / "xpu_profiling_rn50.xml", [PASS, FAIL], failures=1)

    assert not ut_results.Checker(work_dir=tmp_path).run("xpu_profiling").passed


def test_profiling_check_passes_on_a_clean_report(tmp_path):
    write_xml(tmp_path / "xpu_profiling_rn50.xml", [PASS])

    assert ut_results.Checker(work_dir=tmp_path).run("xpu_profiling").passed


# --------------------------------------------------------------------------- #
# known issues
# --------------------------------------------------------------------------- #


def test_known_issue_cases_are_taken_per_category():
    issues = "\n".join([
        "Issue #1: something",
        "Cases:",
        "op_ut,TestFoo,test_a",
        "op_regression,TestFoo,test_b",
        "",
        "Issue #2: other",
        "op_ut,TestFoo,not_in_a_cases_block",
    ])

    assert ut_results.KnownIssues.cases_of(issues, "op_ut") == ["op_ut,TestFoo,test_a"]
    assert ut_results.KnownIssues.cases_of(issues, "op_regression") == \
        ["op_regression,TestFoo,test_b"]


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #


def test_enrich_new_failures_matches_on_category_class_and_test(tmp_path):
    ut_results.write_lines(tmp_path / "ut_failure_list.csv", [
        "| op_regression | TestFoo | test_bad | failed | one | XML |",
        "| op_extended | TestFoo | test_bad | failed | two | XML |",
        "| op_extended | TestFoo | test_bad_longer | failed | three | XML |",
    ])

    rows = ut_results.enrich_new_failures(tmp_path, ["op_extended,TestFoo,test_bad"])

    assert rows == ["| op_extended | TestFoo | test_bad | failed | two | XML |"]


def test_summary_reports_checks_and_gates_one_run(tmp_path, monkeypatch):
    artifact = tmp_path / "artifacts" / "Inductor-XPU-UT-Data-sha-op_regression-1-1"
    artifact.mkdir(parents=True)
    (artifact / "ut_meta.env").write_text(
        "UT_NAME=op_regression\nUT_RUNNER=bmg-test\nUT_CATEGORY=\n", encoding="utf-8")
    write_xml(artifact / "op_regression.xml",
              [PASS] * 352 + [FAIL], tests=353, failures=1)
    monkeypatch.setattr(ut_results.KnownIssues, "_fetch", lambda self, *a: "")

    work = tmp_path / "work"
    code = ut_results.summarize(tmp_path / "artifacts", tmp_path / "none", work, "main")

    assert code == 1
    assert (work / "all_new_failures.txt").read_text() == "op_regression,TestFoo,test_bad\n"
    assert (work / "all_new_failures.csv").read_text() == \
        "| op_regression | TestFoo | test_bad | failed | RuntimeError: boom | XML |\n"
    panel = (work / "ut_summary.md").read_text()
    assert "#### op_regression" in panel
    assert "| op_regression | op_regression | 1 | ❌ failed |" in panel
    assert json.loads((work / "op_regression" / "run_health.jsonl").read_text())["healthy"]


def test_summary_is_green_when_every_failure_is_known(tmp_path, monkeypatch):
    artifact = tmp_path / "artifacts" / "Inductor-XPU-UT-Data-sha-op_regression-1-1"
    artifact.mkdir(parents=True)
    (artifact / "ut_meta.env").write_text(
        "UT_NAME=op_regression\nUT_RUNNER=bmg-test\nUT_CATEGORY=\n", encoding="utf-8")
    write_xml(artifact / "op_regression.xml",
              [PASS] * 352 + [FAIL], tests=353, failures=1)
    monkeypatch.setattr(
        ut_results.KnownIssues, "_fetch",
        lambda self, *a: "Issue #1: x\nCases:\nop_regression,TestFoo,test_bad\n",
    )

    work = tmp_path / "work"
    code = ut_results.summarize(tmp_path / "artifacts", tmp_path / "none", work, "main")

    assert code == 0
    assert (work / "all_new_failures.txt").read_text() == ""
    assert "| op_regression | op_regression | 0 | ✅ passed |" in \
        (work / "ut_summary.md").read_text()


def test_summary_fails_when_no_artifact_names_a_ut_job(tmp_path, monkeypatch):
    (tmp_path / "artifacts" / "junk").mkdir(parents=True)
    monkeypatch.setattr(ut_results.KnownIssues, "_fetch", lambda self, *a: "")

    assert ut_results.summarize(
        tmp_path / "artifacts", tmp_path / "none", tmp_path / "work", "main") == 1
