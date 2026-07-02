---
name: fix/verify
description: >
  Verify a fix by running tests against a source build. Optionally runs a
  before/after comparison and lint. Used by both issue-handler and
  nightly-ci-fix orchestrators after fix/implement.
---

# Verify — Confirm the Fix Works

Runs the test against a source build (with the fix applied) and reports
whether the fix is effective. Always uses source build — never nightly wheel,
since the fix lives in local code.

## Inputs

- `refined_command` — the exact test command from `fix/reproduce` output.
- `pytorch_dir` — path to local PyTorch checkout.
- `changed_files` — list of changed files; if any are `.cpp`/`.h`/`.sycl`,
  a rebuild is required before running.
- `run_before_after_diff` (bool, default `false`) — if `true`, runs the test
  before and after the fix to produce a comparison table. Set to `true` by
  `nightly-ci-fix`.
- `run_lint` (bool, default `false`) — if `true`, runs `spin fixlint` after a
  passing result. Set to `true` by `nightly-ci-fix`.

## Step 1: Confirm source build environment

```bash
python -c "import torch; print(torch.version.git_version)"
```

This must return a commit hash (source build), not a version string like
`2.8.0.dev` with no hash (wheel install). If it is a wheel, stop and report
to the orchestrator — verify requires source build.

Activate the environment before running — load the `/xpu-build-pytorch` skill now.

## Step 2: Rebuild if needed

If any of `changed_files` are C++/SYCL (`.cpp`, `.h`, `.cu`, `.sycl`),
rebuild using the build command from the domain skill loaded by the
orchestrator (e.g. `fix/domains/xpu-kernel`).

Python-only changes (`*.py`) need no rebuild.

## Step 3: Before/after comparison (if run_before_after_diff=true)

**Contract:** this step requires that `fix/implement` left changes staged but
uncommitted. `git stash -u` temporarily removes them to obtain a before-state.
If the orchestrator has already committed the changes before calling verify,
the stash will find nothing and the before recording is skipped.

```bash
# Record BEFORE (without the fix)
git stash -u   # stash staged, unstaged, and untracked changes
# run test, capture output
git stash pop

# Record AFTER (with the fix)
# run test, capture output
```

Output a comparison table:

```
| Test case | Before | After |
|-----------|--------|-------|
| TestFooXPU::test_bar | FAILED (AssertionError: ...) | PASSED |
```

If `git stash -u` reports "No local changes to save", skip the before
recording and only record after.

## Step 4: Run test

Read [../references/run-test.md](../references/run-test.md) now for command
format and result interpretation.

Run ALL failing test cases from the original report individually — do not
assume one representative case is sufficient.

## Step 5: Lint (if run_lint=true)

Only run after a passing test result.

```bash
spin fixlint
spin lint 2>&1 | tail -40
```

- If **clean**: include `lint: clean` in the `PASSED` output.
- If **errors remain after fixlint**: return `FAILED` with the lint errors as
  `failure_output` and `suggestion: lint errors remaining after auto-fix`.

## Output

Return to the orchestrator:

```
PASSED
  before_after_diff: <comparison table, if run_before_after_diff=true>
  lint: clean | <issues fixed>

FAILED
  failure_output: <relevant test output>
  suggestion: <what might need to change in the fix>

CANNOT_VERIFY
  blocker: <what went wrong (env, rebuild failure, 0 collected, timeout)>
```

The orchestrator decides whether to loop back to `fix/implement` on `FAILED`,
or proceed to commit/PR on `PASSED`.
