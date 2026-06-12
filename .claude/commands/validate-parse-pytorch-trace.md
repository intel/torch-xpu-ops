# Validate parse-pytorch-trace Skill

Validate the `parse-pytorch-trace` skill by running its script against test fixtures
with known expected outputs.

## Instructions

1. Load the `parse-pytorch-trace` skill.
2. Read `test/agentic/parse-pytorch-trace/test_cases.md` for the full list of test cases.
3. For each case:
   - Run the script on the specified fixture file using the given options.
   - Check each "Expected" criterion against the actual output.
   - Record whether the criterion was met (CORRECT) or not (FAILED).
4. Output a summary table showing:
   - Case name
   - Criterion
   - Expected result (PASS)
   - Actual result (CORRECT or FAILED)
   - Notes (any discrepancy details)

## Optional argument

If a specific case number is provided as `$ARGUMENTS`, validate only that case
instead of running all cases.

## Goal

This command validates that the parse-pytorch-trace script correctly:
- Parses Chrome trace format JSON
- Links device kernels to host ops via External id
- Merges nested aten:: ops into top-level parents
- Produces accurate per-op GPU time aggregation
- Sorts and reports results correctly
