# AI Reproducer CI Workflow Design

## Goal

For PRs with the `ai_generated` label, enforce that a reproducer test exists in `test/repro/` in valid pytest format, and automatically run it after build. Comment the result on the PR — on failure, instruct Copilot to fix the code.

## Changes

### 1. Lint Check: Reproducer Validation (`pull.yml`)

Add a step at the end of the existing `preci-lint-check` job:

- **Condition**: `contains(github.event.pull_request.labels.*.name, 'ai_generated')`
- **Logic**:
  1. Check if any `test_*.py` files exist under `test/repro/`
  2. Validate each file follows pytest conventions (`def test_...` or `class Test...`)
- **On failure**: Comment `@PR_author` with specific instructions on what needs to be fixed (missing reproducer or invalid pytest format)
- **Non-ai_generated PRs**: Step is skipped entirely
- **Permissions**: `pull-requests: write` added to `preci-lint-check` job for commenting

### 2. Reusable Workflow: `_linux_reproducer.yml`

A standalone reusable workflow (consistent with `_linux_ut.yml` pattern) containing:

**Jobs:**

- **`runner`**: Uses `get-runner` action to obtain runner info (runner_id, user_id, render_id)
- **`reproducer-test`**: Runs the reproducer in a containerized environment

**Inputs:** `runner`, `pytorch`, `torch_xpu_ops`, `python`

**Environment:**
- **Container**: `intelgpu/ubuntu-24.04-lts2:2523.40` (same as UT jobs)
- **Setup**: Uses `linux-testenv` action to create python env, download build wheels, and install pytorch with dependencies

**Steps:**
1. Checkout the repo
2. Prepare test env via `linux-testenv` action
3. Run `pytest ../torch-xpu-ops/test/repro/ -v` from pytorch directory, capture output
4. Comment on PR using `actions/github-script`:
   - **Pass**: Post a success message
   - **Fail**: Post `@copilot` with error output and fix instructions

### 3. Caller in `pull.yml`

```yaml
reproducer-test:
  needs: [linux-build]
  if: contains(labels, 'ai_generated')
  uses: ./.github/workflows/_linux_reproducer.yml
  with:
    runner: pvc_rolling
    pytorch: ...
    torch_xpu_ops: ...
```

### 4. Comment Templates

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

**Lint check failure (comment @PR_author):**
```
@{author} {specific failure reason}

Please ensure your reproducer follows pytest conventions:
- File name: `test/repro/test_<description>.py`
- Contains `def test_...()` functions or `class Test...` classes
- Runnable via `pytest test/repro/`
```

### 5. Permissions

- `preci-lint-check`: `pull-requests: write` for commenting on lint check failure
- `reproducer-test` (caller in `pull.yml`): `issues: write` for commenting test results, consistent with `linux-ut` and `linux-distributed`

### 6. Scope

- Only triggers for PRs with `ai_generated` label
- Does not affect any existing jobs or their behavior
- The `test/repro/` directory does not exist yet; it will be created by AI-generated PRs that include reproducers
