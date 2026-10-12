# Review SKILL Test Cases

Review the files under `test/agentic/pr-review` to validate the pr-review skill.

## Instructions

1. Load the `pr-review` skill.
2. For each file in `test/agentic/pr-review/`, perform a review as if the file were part of a PR submission.
3. For each test case in the file:
   - Identify whether it is marked as "Expected: FLAG" (bad) or "Expected: PASS" (good).
   - Apply the pr-review skill's detection rules (e.g., subgroup reduction anti-pattern).
   - Report whether the skill would correctly flag or pass each case.
4. Output a summary table showing:
   - Case name
   - Expected result (FLAG or PASS)
   - Actual result from review (FLAGGED or PASSED)
   - Verdict (CORRECT or MISSED or FALSE POSITIVE)

## Optional argument

If a specific file path is provided as `$ARGUMENTS`, review only that file instead of scanning the full `test/agentic/pr-review` directory.

## Goal

This command validates that the pr-review skill correctly identifies all anti-patterns (true positives) and does not flag correct code (no false positives).
