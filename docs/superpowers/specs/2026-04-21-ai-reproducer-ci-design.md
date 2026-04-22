# AI Reproducer CI Workflow Design

## Goal

For PRs with the `ai_generated` label, enforce that a reproducer test exists in `test/repro/` and automatically run it after build. Comment the result on the PR — on failure, instruct Copilot to fix the code.

## Changes

All changes are in `.github/workflows/pull.yml`.

### 1. Lint Check: Reproducer Existence Check

Add a step at the end of the existing `preci-lint-check` job:

- **Condition**: `contains(github.event.pull_request.labels.*.name, 'ai_generated')`
- **Logic**: Check if any `test_*.py` files exist under `test/repro/`. If none found, fail with a message indicating the PR is missing a reproducer.
- **Non-ai_generated PRs**: Step is skipped entirely.

### 2. New Job: `reproducer-test`

- **Depends on**: `linux-build`
- **Condition**: `ai_generated` label is present
- **Runner**: `pvc_rolling`
- **Permissions**: `contents: read`, `issues: write` (consistent with existing jobs)
- **Timeout**: 30 minutes

**Steps:**

1. Checkout the repo
2. Run `pytest test/repro/ -v`, capture output
3. Based on exit code, comment on the PR using `gh` CLI:
   - **Pass**: Post a success message
   - **Fail**: Post `@copilot` with error output and fix instructions

### 3. Comment Templates

**Pass:**
```
✅ Reproducer test passed. All tests in `test/repro/` executed successfully.
```

**Fail:**
```
@copilot The reproducer test failed. Please analyze the error output below and fix the code in this PR accordingly.

<details><summary>Reproducer test output</summary>

{pytest output}

</details>

Instructions:
1. Review the test failure output above
2. Identify the root cause of the failure
3. Update the relevant source code to fix the issue
4. Ensure the reproducer in test/repro/ passes after your fix
```

### 4. Permissions

The `reproducer-test` job uses `issues: write` to comment on PRs, matching the pattern used by `linux-ut` and `linux-distributed`.

### 5. Scope

- Only triggers for PRs with `ai_generated` label
- Does not affect any existing jobs or their behavior
- The `test/repro/` directory does not exist yet; it will be created by AI-generated PRs that include reproducers
