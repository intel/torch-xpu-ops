---
name: ut-check
description: Analyze UT (unit test) results for a torch-xpu-ops PR. Use when asked to check test results, analyze CI failures, or evaluate test coverage for a PR. Produces a structured report of new failures, failure relevance, and new test coverage.
---

# UT Result Check Skill

Analyze unit test results for a torch-xpu-ops PR: identify new failures, assess
whether they relate to the PR changes, and verify new test coverage.

## Input

The UT data is provided as a JSON file (typically `/tmp/ut_data.json`) produced
by `.github/scripts/bot_ut_check.py`. The JSON contains:

```json
{
  "pr_number": 1234,
  "run_id": 56789,
  "failures": [
    {"category": "...", "class": "TestFoo", "test": "test_bar", "status": "FAILED", "message": "..."}
  ],
  "changed_files": {
    "operator_source": ["src/ATen/native/xpu/Foo.cpp"],
    "test_files": ["test/xpu/test_foo.py"],
    "skip_lists": [],
    "other": []
  },
  "new_tests": ["TestFoo::test_bar"],
  "new_tests_passed": ["TestFoo::test_bar"],
  "new_tests_not_run": [],
  "passed_tests_count": 12345
}
```

## Analysis Workflow

### Step 1: Read the Data

Read the JSON file. Understand the scope: how many failures, which files the PR
changed, which new tests were added.

### Step 2: Classify Failures

For each new failure, determine relevance to the PR:

- **Related** -- the failure is in a test module that directly tests an operator
  modified by this PR (e.g., PR changes `src/ATen/native/xpu/sycl/FooKernels.cpp`
  and `test_foo_xpu` fails).
- **Possibly related** -- the failure is in a related subsystem or could be
  caused by a side effect of the change.
- **Unrelated** -- the failure is in an unrelated test module and is likely a
  pre-existing flaky test or infrastructure issue.

Use the `changed_files.operator_source` list to map operator files to test
modules. Common mappings:
- `src/ATen/native/xpu/sycl/<Op>Kernels.cpp` -> `test/xpu/test_<op>_xpu.py`
- `src/ATen/native/xpu/<Op>.cpp` -> same test file

### Step 3: Evaluate New Test Coverage

If the PR adds or modifies test files, check:
- Which new tests ran and passed (`new_tests_passed`)
- Which new tests failed (cross-reference with `failures`)
- Which new tests were NOT RUN (`new_tests_not_run`) -- this is a concern

### Step 4: Produce the Report

Follow the output format below. Apply the truncation rules strictly.

## Truncation Rules

When lists are long, showing every item clutters the report without adding value.
Apply these limits:

- **New failures**: If more than 20, show the first 20 in the table, then add:
  `... and N more. See CI logs for the full list.`
- **New/modified tests**: If more than 20, show the first 20 in the table, then
  add: `... and N more new tests (M passed, K failed, J not run).`
- **Changed files**: List individual files only if <= 10 per category. Otherwise
  summarize as counts (e.g., "42 operator source files changed").

Always include the total count so the reader knows the full scope.

## Output Format

```markdown
## UT Result Check: PR #<number>

### New Failures
<count> new failure(s) detected. / No new failures detected.

| Test | Category | Status | Related to PR? |
|------|----------|--------|----------------|
| `ClassName::test_name` | category | FAILED | Related / Possibly related / Unrelated |
...

### Failure Relevance Analysis
Brief explanation of why each failure is classified as it is. Group related
failures together rather than repeating the same reasoning per test.

### New Test Coverage
| New/Modified Test | Status |
|-------------------|--------|
| `ClassName::test_name` | PASSED / FAILED / NOT RUN |
...

### Summary
One-sentence verdict: safe to merge, needs investigation, or blocked by failures.
```

**Omit sections that have no content.** If there are no new failures, omit the
"Failure Relevance Analysis" section. If the PR adds no tests, omit "New Test
Coverage."

If a calling workflow explicitly requires a skill marker, append this exact
literal final line:
Custom skills applied: ut-check.

Otherwise, keep the reply in the requested report format and do not force an
extra trailing sentence.
