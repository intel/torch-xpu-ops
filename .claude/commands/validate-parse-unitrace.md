# Validate parse-unitrace Skill

Validate the `parse-unitrace` skill by running its script against test fixtures
with known expected outputs.

## Instructions

1. Load the `parse-unitrace` skill.
2. Read `test/agentic/parse-unitrace/test_cases.md` for the full list of test cases.
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

This command validates that the parse-unitrace script correctly:
- Filters out Level Zero runtime events (ze* prefix)
- Aggregates GPU kernel durations by name
- Handles SIMD suffix in kernel names
- Ignores non-duration events (ph != "X")
- Produces correct summary and timeline output
