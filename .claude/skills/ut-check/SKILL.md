---
name: ut-check
description: Analyze UT (unit test) results for a torch-xpu-ops PR. Use when asked to check test results, analyze CI failures, or evaluate test coverage for a PR. Produces a structured report of new failures, failure relevance, and new test coverage.
---

# UT Result Check Skill

Analyze unit test results for a torch-xpu-ops PR: identify new failures, assess
whether they relate to the PR changes, and verify new test coverage.

## Quick start

1. Read the UT data JSON at the path given in the request (default
   `/tmp/ut_data.json`). If it is missing or unreadable, say so and stop --
   never reconstruct the data from elsewhere.
2. Analyze it following the workflow below. The data is self-contained; no
   sub-agents or extra CI queries are needed.
3. Emit the report as your final message. The calling workflow posts it as the
   PR comment -- do not post a comment yourself.

## Input

The UT data is provided as a JSON file (typically `/tmp/ut_data.json`) produced
by `.github/scripts/bot_ut_check.py`. The script already de-duplicates failures,
classifies each failure's relevance, summarizes new-test coverage, and computes
a baseline verdict. The JSON contains:

```json
{
  "pr_number": 1234,
  "run_id": 56789,
  "failures": [
    {"category": "op_ut", "class": "a.b.TestFoo", "test": "test_bar",
     "status": "failed", "message": "...", "relevance": "Related"}
  ],
  "changed_files": {
    "operator_source": ["src/ATen/native/xpu/Foo.cpp"],
    "test_files": ["test/xpu/test_foo_xpu.py"],
    "skip_lists": [],
    "other": []
  },
  "new_tests": ["TestFoo::test_bar"],
  "new_tests_summary": {
    "passed": ["TestFoo::test_bar"],
    "failed": [],
    "skipped": [],
    "not_run": []
  },
  "passed_tests_count": 12345,
  "totals": {"test_cases": 0, "passed": 0, "skipped": 0,
             "failures": 0, "errors": 0},
  "verdict": "Safe to merge",
  "verdict_reason": "No new failures detected."
}
```

### Ground Truth for New Failures

The authoritative source for new failures is the `New-UT-Failures-*` artifact
which contains `new_ut_failure_list.csv`. This CSV is produced by the CI
summary job: `ut_result_check.sh` identifies new failures (not in the known
issues list), and the workflow enriches them with error messages.

The `Inductor-XPU-UT-Data-*` artifact bundles a **duplicate** copy of that CSV;
`bot_ut_check.py` reads only the authoritative artifact and de-duplicates by
`(category, class, test)`. If you ever see the same failure listed twice, treat
it as a single failure and note the collection anomaly.

## Analysis Workflow

### Step 1: Read the Data

Read the JSON. Understand the scope: failure count, changed files, new tests.

### Step 2: Review / Refine Failure Relevance

Each failure already carries a deterministic `relevance`
(`Related` / `Possibly related` / `Unrelated`) computed from the changed files.
Trust it as a baseline, but refine using the error `message` and your judgment:

- **Related** -- failure is in a test module that directly tests an operator or
  test file this PR modifies.
- **Possibly related** -- failure is in a related subsystem or a plausible side
  effect of the change.
- **Unrelated** -- failure is in an unrelated module; likely pre-existing flake
  or infrastructure issue (e.g. missing shared library, download error).

If you change a classification, briefly say why.

### Step 3: Evaluate New Test Coverage

Use `new_tests_summary`: `passed`, `failed`, `skipped`, `not_run`. Newly added
tests that FAILED or did NOT RUN are a concern and must be called out.

### Step 4: Produce the Report

Follow the output format below exactly. Apply the truncation rules strictly.

## Guardrails: Do Not Post Low-Quality Comments

- **Never fabricate or guess** counts. Report exactly what the JSON contains.
- If `failures` is empty **and** `new_tests` is empty, produce a concise report
  (New Failures: none; Recommendation) rather than padding with empty tables.
- If the data looks internally inconsistent (e.g. a failure count that seems
  implausible for the diff, or `totals` missing), state the uncertainty plainly
  in the Recommendation instead of asserting a confident verdict.
- Do not invent a "Failure Relevance Analysis" for failures that are all clearly
  unrelated; a one-line grouping is enough.

## Truncation Rules

- **New failures**: If more than 20, show the first 20, then add:
  `... and N more failure(s). See CI logs for the full list.`
- **New/modified tests**: If more than 20, show the first 20, then add:
  `... and N more new test(s). See CI logs for the full list.`
- **Changed files**: Summarize as counts per category.

Always include the total count so the reader knows the full scope.

## Output Format

Every report MUST include: the New Failures section (with the `Related to PR?`
column), a New Test Coverage summary line with counts whenever the PR adds
tests, and a Recommendation with an explicit safe-to-merge verdict.

```markdown
## UT Result Check: PR #<number>

### New Failures
<count> new failure(s) detected (not in known issues). / No new failures detected.

| Test | Category | Status | Related to PR? |
|------|----------|--------|----------------|
| `ClassName::test_name` | category | failed | Related / Possibly related / Unrelated |
...

### Failure Relevance Analysis
Brief explanation, grouping related failures together. Omit if there are no
failures or all are trivially unrelated (a one-line note then suffices).

### New Test Coverage
This PR adds/modifies **<N>** test(s): <p> passed, <f> failed, <s> skipped, <n> not run.

| New/Modified Test | Status |
|-------------------|--------|
| `ClassName::test_name` | PASSED / FAILED / SKIPPED / NOT RUN |
...

### Recommendation
**<Safe to merge | Likely safe to merge | Investigate before merging | Not safe to merge>.**
One-to-two sentence justification grounded in the failures and new-test results.
```

**Omit sections that have no content.** If the PR adds no tests, omit "New Test
Coverage". Always keep the "New Failures" and "Recommendation" sections.

If a calling workflow explicitly requires a skill marker, append this exact
literal final line:
Custom skills applied: ut-check.

Otherwise, keep the reply in the requested report format and do not force an
extra trailing sentence.
