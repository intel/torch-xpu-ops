---
name: fix-verify
description: >
  Use when asked to verify a fix works, confirm a staged patch resolves a
  failure, or produce a before/after summary of a fix.
  Runs the test command against a source build with the fix applied and
  reports PASSED / FAILED / CANNOT_VERIFY. Called by both issue-handler
  and xpu-nightly-ci-fix orchestrators after fix-implement.
---

# Verify — Confirm the Fix Works

Runs the test with the fix applied and reports whether the fix is
effective. A fix that has to be compiled in is verified against a source
build — never a nightly wheel, which cannot see local code.

## Contents

- [Inputs](#inputs)
- [Shell helpers](#shell-helpers)
- Step 1: [Classify whether a rebuild is needed](#step-1-classify-whether-a-rebuild-is-needed)
- Step 2: [Verify the fix](#step-2-verify-the-fix)
- Step 3: [Run test](#step-3-run-test)
- Step 4: [Lint](#step-4-lint)
- [Output](#output)

## Inputs

- `refined_command` — the exact test command from `fix-reproduce`'s
  output.
- `PYTORCH_DIR` — path to local PyTorch checkout.
- `target_repo_dir` — path to the checkout that holds the staged fix
  (same derivation rule as in `fix-implement`: equals
  `PYTORCH_DIR` for `target_repo=pytorch`,
  `<PYTORCH_DIR>/third_party/torch-xpu-ops` for `target_repo=torch-xpu-ops`).
  All git operations run against `target_repo_dir`. The rebuild (Step 2)
  still runs from `PYTORCH_DIR` because `pip install -e .` builds pytorch
  and pulls its submodule pin.
- `changed_files` — list of changed files from `fix-implement`'s
  output; if any are C++/SYCL (`.cpp`, `.h`, `.cu`, `.sycl`) or CMake
  (`CMakeLists.txt`, `*.cmake`), a rebuild is required before running.

This skill always rebuilds if needed (Step 2) and runs the test with the
fix applied (Step 3), then lints a passing result (Step 4); there are no
flags to toggle any of these off.

## Shell helpers

The recipes below call `abort` (exit non-zero with a diagnostic). It is
not a shell builtin — define it once at the top of your shell, same as
the orchestrators do:

```bash
abort() { echo "ABORT: $*" >&2; exit 1; }
```

An `abort` in this skill means "return `CANNOT_VERIFY` to the
orchestrator with that message as the `blocker`", never "continue
silently".

## Step 1: Classify whether a rebuild is needed

This step only classifies; the rebuild itself happens in Step 2.

If any of `changed_files` are C++/SYCL (`.cpp`, `.h`, `.cu`, `.sycl`)
or CMake (`CMakeLists.txt`, `*.cmake`), a rebuild is required so the fix
is compiled in before the test runs. On the first rebuild (Step 2), clean
the build cache and retry once if `xpu-build-pytorch` reports a
cache-related failure. For a torch-xpu-ops fix the rebuild also needs the
`third_party/xpu.txt` pin override that makes CMake see the fix (done in
Step 2).

If all changed files are python-only, no rebuild is needed — nothing has
to be compiled for the edit to take effect.

## Step 2: Verify the fix

Reaching this skill means `fix-reproduce` already ran the test and
observed the failure (the "before" state). This step rebuilds with the
fix applied (when needed) and sets up for the Step 3 test run that
confirms it now passes. `fix-implement` always leaves the fix
**staged**; the caller may additionally have committed it onto a branch
before invoking this skill. Both arrangements are accepted — do not
assume either. No stash/checkout dance is needed.

**Assert the tested tree is the tree that gets handed off.** The caller
either exports `base_sha..branch` (committed) or picks up
`git diff --cached` (staged), so the fix must live entirely in one of
those two places — never partly in the worktree. Before rebuilding:

```bash
# Nothing may be left in the worktree: an unstaged edit would be tested
# here but excluded from both hand-off paths. Check this first, so a fix
# that was edited but never staged reports the precise cause.
git -C "$target_repo_dir" diff --quiet || \
  { echo "FAILED reason=unstaged_changes_present"; exit 1; }
```

`unstaged_changes_present` catches a re-implement retry that edited a
file without staging or amending it, which would make the tested tree
differ from what the caller hands off.

All git commands here run against `target_repo_dir` (not `PYTORCH_DIR`);
these can differ when `target_repo == "torch-xpu-ops"`.

**When any `changed_files` are C++/SYCL or CMake**, the fix must be
compiled in before the test (per Step 1). Python-only changes need no
rebuild.

```bash
# For torch-xpu-ops fixes, point the pin at the working branch so the
# rebuild sees the fix (Commit Pin & Development Override in AGENTS.md;
# do NOT stage or commit this file).
if [ "$target_repo_dir" != "$PYTORCH_DIR" ]; then
    git -C $target_repo_dir rev-parse HEAD > $PYTORCH_DIR/third_party/xpu.txt
fi
# Rebuild WITH the fix (only if C++/SYCL or CMake changed):
#   invoke xpu-build-pytorch skill here
# Then run the test in Step 3; its result is the "after" output.
```

**Confirm the tree under test is the source build.** A fix that the
running interpreter does not pick up cannot be verified. Checking
`torch.version.git_version` is NOT sufficient: released and nightly
wheels also carry a real commit hash. Check where `torch` is imported
from:

```bash
python -c "import torch, os; print(os.path.realpath(torch.__file__))"
```

- **A rebuild was required** (C++/SYCL/CMake): the printed path must be
  under `$(realpath $PYTORCH_DIR)/torch/` — the rebuild above is what
  produces that source build. If it still resolves into `site-packages`
  of an unrelated prefix, the rebuild did not take effect: return
  `CANNOT_VERIFY(reason=wheel_install_not_source)` with `blocker="torch
  imported from <path>; verify requires a source build"`.
- **No rebuild was required** (python-only): a wheel install is fine for
  files the interpreter reads from the checkout (test files, skip
  lists), which is where python-only fixes normally live. A python fix
  *inside* the installed package (`torch/`) is not picked up by a wheel
  — same `CANNOT_VERIFY(reason=wheel_install_not_source)`.

The "before" cell of the comparison table (Output section) is filled
from `fix-reproduce`'s recorded failure, not re-run here; "after" is the
result from Step 3 with the fix applied. The two come from different
phases (reproduce may have run against a nightly wheel, verify against
the rebuilt tree), so the table is a before-fix-vs-after-fix
summary, not a single-build A/B.

> **Caveat (read before trusting a PASSED verdict):** because the
> "before" is a nightly-wheel failure and the "after" is a source build
> with the fix, a PASSED verdict means "the test passes on a source build
> that includes the fix" — it does **not** prove the fix is what made it
> pass. If the bug was already resolved upstream after the nightly was
> cut, the source build passes regardless of the fix. This skill does not
> re-run the source build without the fix to rule that out (that would
> cost a second cold build). State this limitation in the report; a human
> reviewing the patch should confirm the change is actually responsible.

## Step 3: Run test

Run ALL failing test cases from the original report individually.

Result interpretation:

- `all skipped` → read pytest's `SKIPPED [N] <file:line>: <reason>`
  line to classify:
  - Reason matches an XPU marker (`skipIfXpu`, `xfailIfXPU`,
    `expectedFailureXPU`, `skipXPU`, or a message containing "xpu")
    → `FAILED` with `reason=stale_skip_after_fix`. Rationale: a fix
    that leaves the failing test skipped is an incomplete fix. This
    is intentionally different from `fix-reproduce`'s handling —
    reproduce temporarily unskips to confirm the bug, but verify's
    job is to confirm the fix; if the XPU-marker skip is still in
    place, the fix did not touch what it should have. Suggest in
    `reason_detail`: "test still skipped after fix — the stale skip
    decorator should have been removed as part of the fix (see the
    Skip operations section of `fix-implement`)."
  - Reason is environmental (missing dep, no GPU, tool not found,
    "no accelerator", etc.) →
    `CANNOT_VERIFY(reason=environmental_skip)` with
    `blocker="test skipped for environmental reason: <marker>"`.
- `xfailed` → `FAILED` with `reason=xfail_after_fix`.
- `FAILED` → `FAILED` with `reason=test_still_failing`.
- `PASSED` → `PASSED` with `reason=ok`.

## Step 4: Lint

Always run after a passing test result. Run it in `target_repo_dir` —
the repo that owns the changed files — not in `PYTORCH_DIR`:

```bash
cd $target_repo_dir
spin fixlint
# fixlint rewrites files in the working tree, leaving them unstaged.
# Stage them so the lint fixes travel with the fix and Step 2's
# clean-worktree invariant holds; the caller folds the staged lint fixes
# into its hand-off (issue-handler amends them into the fix commit).
# Nothing outside changed_files may be folded in; if `git status` shows
# fixlint touched other files, return FAILED rather than widening the diff.
git -C $target_repo_dir add -- <changed_files>
spin lint 2>&1 | tail -40
```

- If **clean**: include `lint: clean` in the `PASSED` output.
- If **errors remain after fixlint**: return
  `FAILED(reason=lint_errors_after_autofix)` with the lint errors as
  `failure_output` and suggest that the remaining errors need a
  human touch.

## Output

Return to the orchestrator a report (markdown block plus JSON block).
The skill does not commit or push — the caller consumes stdout and
decides what to do, per the pattern established by `issue-triage`,
`fix-reproduce`, `fix-root-cause`, and `fix-implement`.

Include the `<!-- agent:verify -->` marker on the first line of the
markdown block so a downstream caller can locate its own previous
verify comment (if any) and update it in place. Comment location and
update is the **caller's** responsibility.

```
<!-- agent:verify -->

## Verify Result

- **Target repo:** <pytorch | torch-xpu-ops>
- **Refined command:** <refined_command>
- **Verdict:** <PASSED | FAILED | CANNOT_VERIFY> — <one-line reason>
- **Lint:** <clean | errors: <summary>>

### Before / after

Before = the failure `fix-reproduce` recorded; After = the Step 3 test
result with the fix applied.

| Test case | Before | After |
|-----------|--------|-------|
| TestFooXPU::test_bar | FAILED (AssertionError: ...) | PASSED |

*Automated by fix-verify.*
```

```json
{
  "target_repo": "pytorch or torch-xpu-ops",
  "refined_command": "<echo of input>",
  "changed_files": ["path/to/file1.py"],
  "verdict": "PASSED or FAILED or CANNOT_VERIFY",
  "reason": "<enumerated reason code, see below>",
  "reason_detail": "one-line human-readable detail",
  "before_after_table": "<markdown table string, or null>",
  "failure_output": "<test/lint output excerpt on FAILED, or null>"
}
```

### Field contract

- `target_repo` — echo from `fix-implement`'s output.
- `refined_command` — echo the exact command that was run.
- `changed_files` — echo from `fix-implement`; downstream orchestrator
  uses this to build the commit's file list.
- `before_after_table` — the markdown table pairing `fix-reproduce`'s
  recorded failure (before) with the Step 3 result (after); `null` only
  when the test never ran (e.g. a `CANNOT_VERIFY` before Step 3).
- `failure_output` — non-null on `FAILED`; excerpt of the test or lint
  output (bounded — do not dump multi-MB logs, ~40 lines is enough for
  a human to see the failure).

### `reason` values

On `verdict=PASSED`:

- `ok` — test passed with the fix applied; lint clean.

On `verdict=FAILED`:

- `test_still_failing` — Step 3 reported `FAILED`; fix is incomplete.
- `unstaged_changes_present` — Step 2 found edits left in the worktree,
  outside both hand-off paths; the tested tree would differ from what
  the caller picks up.
- `stale_skip_after_fix` — Step 3 reported `all skipped` with an XPU
  marker still in place.
- `xfail_after_fix` — Step 3 reported `xfailed`; fix did not turn the
  test green.
- `lint_errors_after_autofix` — Step 4 could not clean the lint after
  `spin fixlint`; a human touch is needed.
- `unrelated_files_touched_by_lint` — Step 4's `spin fixlint` modified
  files outside `changed_files`.

On `verdict=CANNOT_VERIFY`:

- `wheel_install_not_source` — Step 2 saw `torch` imported from
  `site-packages`; can't verify a source-tree fix through a wheel.
- `rebuild_failed` — `xpu-build-pytorch` returned failure during the
  Step 2 rebuild.
- `environmental_skip` — Step 3's `all skipped` had an environmental
  reason (missing dep, no accelerator).
- `test_collect_zero` — Step 3's `refined_command` resolves to zero
  collected tests; the fix cannot be validated against the reported
  reproducer.
- `test_timeout` — the test process exceeded its timeout and was
  killed.
- `other` — fallback; put full explanation in `reason_detail`.

The orchestrator decides whether to loop back to `fix-implement` on
`FAILED`, or proceed to commit/PR on `PASSED`, based on `verdict` and
`reason`.
