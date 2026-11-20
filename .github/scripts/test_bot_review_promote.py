# Copyright 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0

"""Tests for bot_review_promote.py: the CI and maintainer rules of the review gate."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bot_review_promote as p  # noqa: E402

A = "op_extended,a.TestA,test_a"
B = "op_extended,a.TestA,test_b"
C = "op_extended,a.TestA,test_c"


def job(name, conclusion, status="completed"):
    return {"name": name, "conclusion": conclusion, "status": status}


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def ut_check(root, categories, new=(), unhealthy=()):
    d = root / "UT-Check-linux-basic-1" / "basic"
    write(d / "verdict.json", json.dumps({"categories": [
        {"category": c, "passed": ok} for c, ok in categories]}))
    write(d / "new_failures.txt", "\n".join(new))
    write(d / "run_health.jsonl", "\n".join(
        json.dumps({"category": c, "healthy": c not in unhealthy}) for c, _ in categories))
    return root


def test_parse_started():
    assert p.parse_started("No blocking issues (review started 2026-11-20T07:00:00Z)") == "2026-11-20T07:00:00Z"
    assert p.parse_started("No blocking issues") is None


def test_classify_against_nightly():
    got = p.classify([A, B, C], nightly_failed={A}, nightly_passed={B})
    assert got == {A: "broken_trunk", B: "new", C: "unknown"}


def test_ut_baseline_reads_case_logs(tmp_path):
    write(tmp_path / "x" / "basic" / "cases" / "failures_op_extended.log", A + "\n")
    write(tmp_path / "x" / "basic" / "cases" / "passed_op_extended.log", B + "\n\n")
    assert p.ut_baseline(tmp_path) == ({A}, {B})


def test_ut_group_tolerates_only_broken_trunk(tmp_path):
    root = ut_check(tmp_path, [("op_extended", False)], new=[A, B])
    blockers, tolerated = p.ut_group_blockers(root, nightly_failed={A}, nightly_passed={B})
    assert tolerated == [A]
    assert len(blockers) == 1 and "test_b" in blockers[0] and "passes on main" in blockers[0]


def test_ut_group_all_broken_trunk_passes(tmp_path):
    root = ut_check(tmp_path, [("op_extended", False), ("op_regression", True)], new=[A])
    assert p.ut_group_blockers(root, {A}, set()) == ([], [A])


def test_ut_group_unhealthy_or_unexplained_blocks(tmp_path):
    root = ut_check(tmp_path / "1", [("op_extended", False)], new=[A], unhealthy={"op_extended"})
    assert "did not run completely" in p.ut_group_blockers(root, {A}, set())[0][0]
    root = ut_check(tmp_path / "2", [("xpu_profiling", False)])
    assert "without a new-failure list" in p.ut_group_blockers(root, set(), set())[0][0]
    assert p.ut_group_blockers(tmp_path / "missing", set(), set(), "UT `x`")[0] == ["UT `x` check results are missing"]


def test_e2e_compares_models_with_nightly(tmp_path):
    header = "dev,backend,suite,dtype,mode,name,scenario,batch_size,elapsed,status,accuracy\n"
    write(tmp_path / "pr" / "E2E-Data-1" / "x-accuracy.csv", header
          + "xpu,d,timm,bf16,training,m1,accuracy,8,1,pass,fail_accuracy\n"
          + "xpu,d,timm,bf16,training,m2,accuracy,8,1,pass,fail_accuracy\n"
          + "xpu,d,timm,bf16,training,m3,accuracy,8,1,pass,pass\n"
          + "xpu,d,timm,bf16,training,m4,accuracy,8,1,pass,timeout\n")
    write(tmp_path / "main" / "E2E-Data-1" / "y-accuracy.csv", header
          + "xpu,d,timm,bf16,training,m1,accuracy,8,1,pass,fail_accuracy\n"
          + "xpu,d,timm,bf16,training,m2,accuracy,8,1,pass,pass\n")
    blockers, tolerated = p.e2e_blockers(p.e2e_accuracy(tmp_path / "pr"), p.e2e_accuracy(tmp_path / "main"))
    assert tolerated == ["timm/bf16/training/m1"]
    assert [b.split("`")[1] for b in blockers] == ["timm/bf16/training/m2", "timm/bf16/training/m4"]
    assert "not run on main" in blockers[1]


def no_call(*_):
    raise AssertionError("must not be called")


def test_ci_waits_for_running_jobs():
    assert p.ci_verdict([job("preci-lint-check", None, "in_progress")], no_call, no_call) == (True, [], [])


def test_ci_green_and_skipped_jobs_pass():
    jobs = [job("preci-lint-check", "success"), job("linux-build (build) / main", "success"),
            job("linux-ut (basic) / test-in-container", "success"),
            job("linux-ut (basic) / check", "success"), job("windows-ut", "skipped")]
    assert p.ci_verdict(jobs, no_call, no_call) == (False, [], [])


def test_ci_failed_build_or_lint_blocks():
    jobs = [job("preci-lint-check", "failure"), job("linux-build (build) / main", "cancelled")]
    _, blockers, _ = p.ci_verdict(jobs, no_call, no_call)
    assert blockers == ["job `preci-lint-check` failure", "job `linux-build (build) / main` cancelled"]


def test_ci_failed_ut_check_is_classified():
    seen = []

    def ut_group(platform, ut):
        seen.append((platform, ut))
        return [], [A]

    jobs = [job("linux-ut (op_ut) / test-in-container", "failure"),
            job("linux-ut (op_ut) / check", "failure"),
            job("linux-distributed (xpu_distributed) / check", "success"),
            job("ut-report", "failure")]
    assert p.ci_verdict(jobs, ut_group, no_call) == (False, [], [A])
    assert seen == [("linux", "op_ut")]


def test_ci_ut_group_without_check_blocks():
    jobs = [job("linux-ut (basic) / test-in-container", "failure"),
            job("linux-ut (basic) / check", "skipped")]
    assert p.ci_verdict(jobs, no_call, no_call)[1] == ["UT `linux-basic` did not finish"]


def test_ci_e2e_summary_failure_is_classified():
    jobs = [job("linux-e2e (timm_models, accuracy) / run", "success"),
            job("linux-e2e-summary / summary", "failure")]
    assert p.ci_verdict(jobs, no_call, lambda: (["E2E `m2`"], ["m1"])) == (False, ["E2E `m2`"], ["m1"])
    jobs = [job("linux-e2e (timm_models, accuracy) / run", "failure"),
            job("linux-e2e-summary / summary", "success")]
    assert p.ci_verdict(jobs, no_call, no_call)[1] == ["job `linux-e2e (timm_models, accuracy) / run` failure"]


def test_maintainer_activity_since_review():
    activity = [
        {"login": "maint", "type": "User", "at": "2026-11-20T08:00:00Z"},
        {"login": "old", "type": "User", "at": "2026-11-20T06:00:00Z"},
        {"login": "author", "type": "User", "at": "2026-11-20T08:00:00Z"},
        {"login": "torchxpubot", "type": "User", "at": "2026-11-20T08:00:00Z"},
        {"login": "ci[bot]", "type": "Bot", "at": "2026-11-20T08:00:00Z"},
        {"login": "outsider", "type": "User", "at": "2026-11-20T08:00:00Z"},
        {"login": "maint2", "type": "User", "at": "2026-11-20T08:00:00Z", "body": " @torchxpubot rerun failed"},
    ]
    roles = {"maint": "write", "maint2": "write", "old": "admin", "author": "admin", "outsider": "read"}
    got = p.maintainer_logins_since(activity, "2026-11-20T07:00:00Z", "author", lambda u: roles.get(u, ""))
    assert got == ["maint"]


def run_promote(monkeypatch, labels, status, ci, activity=()):
    calls = []
    pr = {"number": 7, "state": "open", "draft": False, "labels": [{"name": n} for n in labels],
          "head": {"sha": "s" * 40}, "user": {"login": "author"}, "body": "Reviewers: @alice"}
    monkeypatch.setattr(p, "api", lambda path: pr if "/pulls/" in path else {"statuses": status})
    monkeypatch.setattr(p, "api_items", lambda path, jq=".[]": [])
    monkeypatch.setattr(p, "gh", lambda *a, **k: calls.append(a) or "")
    monkeypatch.setattr(p, "evaluate_ci", lambda repo, sha, d: ci)

    class FakeRepo(p.Repo):
        def activity(self, pr_number, since):
            return list(activity)

        def role(self, login):
            return "write"

    p.promote(FakeRepo("o/r"), 7)
    return calls


PASSED = [{"context": "bot-review", "state": "success",
           "description": "No blocking issues (review started 2026-11-20T07:00:00Z)"}]


def test_promote_swaps_labels_when_review_and_ci_pass(monkeypatch):
    calls = run_promote(monkeypatch, ["in progress"], PASSED, (False, [], ["x"]))
    assert ("api", "repos/o/r/issues/7/labels", "-f", "labels[]=ready for review") in calls
    assert ("api", "-X", "DELETE", "repos/o/r/issues/7/labels/in%20progress") in calls
    assert ("api", "repos/o/r/pulls/7/requested_reviewers", "-f", "reviewers[]=alice") in calls
    assert "Requested review from @alice" in calls[-1][-1]
    assert "also fail on main" in calls[-1][-1] and "bot-review-promote:promoted:" in calls[-1][-1]


def test_promote_holds_on_ci_review_or_maintainer(monkeypatch):
    blocked = run_promote(monkeypatch, ["in progress"], PASSED, (False, ["job `x` failure"], []))
    assert len(blocked) == 1 and "CI blocks" in blocked[0][-1]          # one explanatory comment
    other = run_promote(monkeypatch, ["in progress"], PASSED, (False, ["job `y` failure"], []))
    assert blocked[0][-1].split("-->")[0] != other[0][-1].split("-->")[0]   # new reasons, new marker
    assert run_promote(monkeypatch, ["in progress"], PASSED, (True, [], [])) == []      # CI running
    failed = [{"context": "bot-review", "state": "failure", "description": ""}]
    assert run_promote(monkeypatch, ["in progress"], failed, (False, [], [])) == []
    assert run_promote(monkeypatch, ["ready for review"], PASSED, (False, [], [])) == []
    late = [{"login": "maint", "type": "User", "at": "2026-11-20T08:00:00Z"}]
    held = run_promote(monkeypatch, ["in progress"], PASSED, (False, [], []), late)
    assert len(held) == 1 and "@maint commented after" in held[0][-1]


def test_only_the_latest_attempt_counts(tmp_path):
    def check(attempt, new):
        d = tmp_path / f"UT-Check-linux-basic-{attempt}" / "basic"
        write(d / "verdict.json", json.dumps({"categories": [{"category": "op_extended", "passed": False}]}))
        write(d / "new_failures.txt", new)

    check(1, B)  # stale attempt: B passes on main, would block
    check(2, A)
    assert [d.name for d in p.latest_attempts(tmp_path)] == ["UT-Check-linux-basic-2"]
    assert p.ut_group_blockers(tmp_path, {A}, {B}) == ([], [A])


def test_reviewers_from_description_and_change_requests():
    body = "Fix foo.\n\nReviewers: @alice, bob @intel/xpu-team @Alice author\nreviewer: @carol."
    reviews = [{"user": {"login": "dave"}, "state": "CHANGES_REQUESTED"},
               {"user": {"login": "erin"}, "state": "CHANGES_REQUESTED"},
               {"user": {"login": "erin"}, "state": "APPROVED"},
               {"user": {"login": "dave"}, "state": "COMMENTED"}]
    assert p.reviewers_to_request(body, reviews, "author") == ["alice", "bob", "carol", "dave"]
    assert p.reviewers_to_request(None, [], "author") == []


def test_nightly_baseline_is_at_most_five_days_old():
    from datetime import datetime, timezone
    now = datetime(2026, 11, 20, 12, tzinfo=timezone.utc)
    runs = [{"id": 1, "display_title": "Nightly / Build-from-source", "created_at": "2026-11-19T13:15:00Z"},
            {"id": 2, "display_title": "Nightly / Nightly-wheel", "created_at": "2026-11-19T13:36:00Z"},
            {"id": 3, "display_title": "Weekly / Build-from-source", "created_at": "2026-11-15T16:10:00Z"},
            {"id": 4, "display_title": "Nightly / Build-from-source", "created_at": "2026-11-12T13:15:00Z"}]
    assert [r["id"] for r in p.recent_nightlies(runs, now)] == [1, 3]


def test_request_reviewers_is_capped(monkeypatch):
    calls = []
    body = "Reviewers: " + " ".join(f"@u{i}" for i in range(9))
    monkeypatch.setattr(p, "api_items", lambda path, jq=".[]": [])
    monkeypatch.setattr(p, "gh", lambda *a, **k: calls.append(a) or "")

    class FakeRepo(p.Repo):
        def role(self, login):
            return "write"

    asked = p.request_reviewers(FakeRepo("o/r"), {"number": 7, "body": body, "user": {"login": "a"}})
    assert asked == [f"u{i}" for i in range(p.MAX_REVIEWERS)]
