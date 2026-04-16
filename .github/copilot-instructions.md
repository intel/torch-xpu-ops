# Copilot Instructions for Torch XPU Operators

This repository implements Torch XPU Operators for Intel GPU support.

Repository layout:
- `src/` contains XPU operator implementations and backend-specific code.
- `yaml/` contains operator/config definitions and generated-input sources.
- `test/` contains regression, unit, and behavior validation.
- `.ci/` contains validation and CI-related workflows.
- `tools/` contains support scripts and utilities.

When reviewing pull requests in this repository:

## Primary review priorities
- Prioritize operator correctness, backend-specific behavior, and regression risk.
- Check whether changes in `yaml/` are reflected consistently in implementation and tests.
- Check whether changes in `src/` have focused tests in `test/`.
- Flag risky changes that may silently change dtype handling, indexing behavior, memory semantics, synchronization behavior, or dispatch behavior.
- Prefer concrete review comments that point to files, operators, or missing coverage.

## Review output style
Always separate findings into:
1. Must fix
2. Risks / follow-up
3. Nice to have

## Testing expectations
- Prefer targeted regression tests over broad, generic test changes.
- If implementation changes but no targeted test is added or updated, call it out.
- If a PR only changes tests, check whether the tests match the intended operator behavior, boundary conditions, and failure mode.

## YAML / operator definition review
When a PR changes files under `yaml/`:
- Check whether schema or definition changes are intentional.
- Check whether implementation and tests are updated consistently.
- Flag changes that appear to update definitions without corresponding validation.

## Implementation review
When a PR changes files under `src/`:
- Check operator semantics.
- Check boundary cases.
- Check dtype-specific behavior.
- Check indexing assumptions, shape handling, and layout-sensitive behavior.
- Check whether any fallback, synchronization, or memory-related behavior changed.

## CI / workflow review
When a PR changes `.ci/` or related files:
- Call out validation coverage changes.
- Identify whether test scope has become narrower, less deterministic, or riskier.
- Flag removal or weakening of checks without clear justification.

## Review tone
- Be specific and actionable.
- Prefer correctness and regression-risk comments over style nitpicks.
- Ask clarifying questions when intent is unclear.
``