# Validate map-kernels-to-ops Skill

Validate the `map-kernels-to-ops` skill by running its script against test fixtures
with known expected outputs.

## Instructions

1. Load the `map-kernels-to-ops` skill.
2. Read `test/agentic/map-kernels-to-ops/test_cases.md` for the full list of test cases.
3. For each case:
   - Run the script on the specified fixture files using the given options.
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

This command validates that the map-kernels-to-ops script correctly:
- Matches trace.json kernels with unitrace kernels by execution order
- Strips SIMD suffix from unitrace names for name verification
- Attributes unitrace durations to top-level aten:: ops via External id
- Detects and reports kernel name mismatches
- Produces accurate per-op unitrace GPU time aggregation
