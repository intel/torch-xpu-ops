# Copyright 2020-2025 Intel Corporation
# Licensed under the Apache License, Version 2.0

# pyright: reportUnusedImport=false, reportMissingParameterType=false

import argparse
import json
import sys

from benchmarks import set_benchmark_models
from classifiers import (
    check_platform_specific,
    classify_issue_type,
    classify_issue_type_canonical,
    classify_module,
    classify_test_module,
    extract_os,
    extract_platform,
    generate_summary,
    get_dependency_from_body,
)
from github import fetch_issue, fetch_project_and_type, parse_issue_ref, rest_to_core
from testcases import (
    dedup_test_cases,
    is_unittest_issue,
    parse_e2e_info,
    parse_test_cases_from_body,
    test_case_source,
)
from text import extract_pr_link, extract_reproduce_steps, extract_traceback
from text import _PR_CONTEXT_SIGNALS_RE


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract basic information for one torch-xpu-ops issue as JSON."
    )
    parser.add_argument("issue", help="Issue number or GitHub issue URL")
    parser.add_argument("--repo", help="owner/name for a bare issue number (default intel/torch-xpu-ops); ignored when a full URL is given")
    parser.add_argument("--output", help="Optional path to also write the JSON output")
    parser.add_argument("--pytorch-folder", help="Local pytorch checkout root; used to load authoritative .ci/benchmarks model lists")
    args = parser.parse_args(argv)

    set_benchmark_models(args.pytorch_folder)

    default_owner, default_repo = "intel", "torch-xpu-ops"
    if args.repo:
        if "/" not in args.repo:
            print(f"Invalid --repo value: {args.repo!r} (expected owner/name)", file=sys.stderr)
            sys.exit(2)
        default_owner, default_repo = args.repo.split("/", 1)

    try:
        owner, repo, number = parse_issue_ref(args.issue, default_owner, default_repo)
    except ValueError as err:
        print(err, file=sys.stderr)
        sys.exit(2)

    # fetch_issue raises RuntimeError (fatal) or SystemExit (PR guard). Let the
    # SystemExit propagate; it carries a message and exits nonzero.
    try:
        issue = fetch_issue(owner, repo, number)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    core = rest_to_core(issue)

    # Classifiers run on the RAW issue fields (labels is the list of dicts).
    body = issue.get("body") or ""
    title = issue.get("title") or ""
    labels = issue.get("labels") or []

    summary = generate_summary(body, title)
    itype = classify_issue_type(body, title, labels)
    module = classify_module(body, title, labels)
    test_module = classify_test_module(body, title, labels)
    dependency = get_dependency_from_body(body, labels)

    pt = fetch_project_and_type(owner, repo, number)

    issue_type = classify_issue_type_canonical(pt["github_type"], itype, labels)

    # Build the test_cases list (all cases in the issue). For e2e issues use
    # the benchmark/model extractor; otherwise use the unit-test parser.
    if test_module == "e2e":
        test_cases = parse_e2e_info(body, title)
    else:
        test_cases = parse_test_cases_from_body(body)
    test_cases = dedup_test_cases(test_cases)

    # Tag each unit-test case with its source repo. A test file ending in
    # _xpu belongs to torch-xpu-ops; otherwise it is an upstream pytorch test.
    # E2E cases (dicts with a "benchmark" key) are not tagged.
    for tc in test_cases:
        if "benchmark" not in tc:
            tc["source"] = test_case_source(tc.get("test_file", ""))

    reproduce_steps = extract_reproduce_steps(body, title)
    traceback = extract_traceback(body)
    os_name = extract_os(body)
    platform = extract_platform(body, title, labels)
    platform_specific = check_platform_specific(platform)
    pr_link = extract_pr_link(body, title)

    # Primary unit-test case: first UT-shape case (dict without a "benchmark"
    # key). Top-level test_file/test_class/test_case mirror it for convenience;
    # the full list remains in test_cases.
    primary_tf = primary_tc_class = primary_tc_case = ""
    for tc in test_cases:
        if "benchmark" not in tc:
            primary_tf = tc.get("test_file", "")
            primary_tc_class = tc.get("test_class", "")
            primary_tc_case = tc.get("test_case", "")
            break

    unittest_issue = is_unittest_issue(body, title, labels, test_cases)

    result = {
        "issue_id": core["issue_id"],
        "repo": f"{owner}/{repo}",
        "title": core["title"],
        "body": body,
        "status": core["status"],
        "assignee": core["assignee"],
        "reporter": core["reporter"],
        "labels": core["labels"],
        "created_time": core["created_time"],
        "updated_time": core["updated_time"],
        "milestone": core["milestone"],
        "summary": summary,
        "type": itype,
        "issue_type": issue_type,
        "github_type": pt["github_type"],
        "module": module,
        "test_module": test_module,
        "dependency": dependency,
        "priority": pt["priority"],
        "pytorchxpu_status": pt["project_status"],
        "pytorchxpu_estimate": pt["project_estimate"],
        "pytorchxpu_depending": pt["project_depending"],
        "pytorchxpu_short_comments": pt["project_short_comments"],
        "os": os_name,
        "platform": platform,
        "platform_specific": platform_specific,
        "traceback": traceback,
        "reproduce_steps": reproduce_steps,
        "test_file": primary_tf,
        "test_class": primary_tc_class,
        "test_case": primary_tc_case,
        "test_cases": test_cases,
        "pr_link": pr_link,
    }

    low_confidence = []
    # reproduce_steps: flag when no shell command found, UNLESS this is a unit
    # test issue (the test_file/test_case itself is the reproducer).
    if not reproduce_steps and not unittest_issue:
        low_confidence.append("reproduce_steps")
    # test_cases: flag when none parsed but the issue looks test-related.
    if not test_cases and test_module in ("ut", "e2e"):
        low_confidence.append("test_cases")
    # pr_link: flag when regex found no PR but the body has PR/branch signals.
    if not pr_link and _PR_CONTEXT_SIGNALS_RE.search(body or ""):
        low_confidence.append("pr_link")
    result["low_confidence"] = low_confidence

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()
