---
name: fix-verify
description: >
  Use when asked to verify a fix works, confirm a staged or committed
  patch resolves a failure, or produce a before/after summary of a fix.
  Runs the test command against a source build with the fix applied and
  reports PASSED / FAILED / CANNOT_VERIFY. Called by both issue-handler
  and xpu-nightly-ci-fix orchestrators after fix-implement.
---

# Verify — Confirm the Fix Works

Runs the test against a source build (with the fix applied) and reports
whether the fix is effective. Always uses source build — never nightly
wheel — since the fix lives in local code that a wheel install cannot
see.

## Contents

- [Inputs](#inputs)
- [Shell helpers](#shell-helpers)
- Step 1: [Confirm source build environment](#step-1-confirm-source-build-environment)
- Step 2: [Rebuild if needed](#step-2-rebuild-if-needed)
- Step 3: [Verify the fix](#step-3-verify-the-fix)
- Step 4: [Run test](#step-4-run-test)
- Step 5: [Lint](#step-5-lint)
- [Output](#output)

## Inputs

- `refined_command` — the exact test command from `fix-reproduce`'s
  output.
- `PYTORCH_DIR` — path to local PyTorch checkout.
- `target_repo_dir` — path to the checkout that holds the fix (staged or
  committed)
  (same derivation rule as in `fix-implement`: equals `PYTORCH_DIR` for
  `target_repo=pytorch`, `<PYTORCH_DIR>/third_party/torch-xpu-ops` for
  `target_repo=torch-xpu-ops`). All git operations run against
  `target_repo_dir`. The rebuild (Step 3) still
  runs from `PYTORCH_DIR` because `pip install -e .` builds pytorch and
  pulls its submodule pin.
- `changed_files` — list of changed files from `fix-implement`'s
  output; if any are C++/SYCL (`.cpp`, `.h`, `.cu`, `.sycl`) or CMake
  (`CMakeLists.txt`, `*.cmake`), a rebuild is required before running.

This skill always runs the test with the fix applied (Step 3) and lints
a passing result (Step 5); there are no flags to toggle either off.

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

## Step 1: Confirm source build environment

The installed `torch` must be the one built from `PYTORCH_DIR`, not a
wheel. Checking `torch.version.git_version` is NOT sufficient —
released and nightly wheels also carry a real commit hash there.
Check where `torch` is imported from instead:

```bash
python -c "import torch, os; print(os.path.realpath(torch.__file__))"
```

The printed path must be under `$(realpath $PYTORCH_DIR)/torch/`. If it
resolves into `site-packages` of an unrelated prefix, the environment
is a wheel install: return
`CANNOT_VERIFY(reason=wheel_install_not_source)` with `blocker="torch
imported from <path>; verify requires a source build"` — a fix staged
in the local tree has no effect on an installed wheel. Producing the
source build is the orchestrator's job (both orchestrators load
`xpu-build-pytorch` before calling this skill when `fix-reproduce`
reproduced at `stage=nightly`).

## Step 2: Rebuild if needed

If any of `changed_files` are C++/SYCL (`.cpp`, `.h`, `.cu`, `.sycl`)
or CMake (`CMakeLists.txt`, `*.cmake`), a rebuild is required so the fix
is compiled in before the test runs. On the first rebuild, clean the
build cache and retry once if `xpu-build-pytorch` reports a cache-related
failure. The rebuild itself (and, for a torch-xpu-ops fix, the
`third_party/xpu.txt` pin override that makes CMake see the fix) happens
in Step 3.

If all changed files are python-only, no rebuild is needed — the source
tree already reflects the edit.

## Step 3: Verify the fix

Reaching this skill means `fix-reproduce` already ran the test and
observed the failure (the "before" state). This step only needs to run
the test **with the fix applied** and confirm it now passes; the fix may
be staged or already committed — either is fine, no stash/checkout dance.

All git commands here run against `target_repo_dir` (not `PYTORCH_DIR`);
these can differ when `target_repo == "torch-xpu-ops"`.

**When any `changed_files` are C++/SYCL or CMake**, the fix must be
compiled in before the test (per Step 2). For python-only changes the
installed source tree already reflects the edit.

```bash
# For torch-xpu-ops fixes, point the pin at the working branch so the
# rebuild sees the fix (Commit Pin & Development Override in AGENTS.md;
# do NOT stage or commit this file).
if [ "$target_repo_dir" != "$PYTORCH_DIR" ]; then
    git -C $target_repo_dir rev-parse HEAD > $PYTORCH_DIR/third_party/xpu.txt
fi
# Rebuild WITH the fix (only if C++/SYCL or CMake changed):
#   invoke xpu-build-pytorch skill here
# Then run the test in Step 4; its result is the "after" output.
```

The "before" cell of the comparison table (Output section) is filled
from `fix-reproduce`'s recorded failure, not re-run here; "after" is the
result from Step 4 with the fix applied. The two come from different
phases (reproduce may have run against a nightly wheel, verify always
against a source build), so the table is a before-fix-vs-after-fix
summary, not a single-build A/B.

## Step 4: Run test

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

## Step 5: Lint

Always run after a passing test result. Run it in `target_repo_dir` —
the repo that owns the changed files — not in `PYTORCH_DIR`:

```bash
cd $target_repo_dir
spin fixlint
# fixlint rewrites files in the working tree, which UNSTAGES those
# hunks. The orchestrator commits what is staged, so re-stage the
# same files or the lint fixes are silently dropped from the commit.
git -C $target_repo_dir add -- <changed_files>
# Nothing outside changed_files may become staged; if `git status`
# shows fixlint touched other files, return FAILED rather than
# widening the diff.
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

Before = the failure `fix-reproduce` recorded; After = the result from
Step 3 with the fix applied.

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
  recorded failure (before) with Step 3's result (after); `null` only
  when Step 3 could not produce an after result (e.g. a `CANNOT_VERIFY`
  before the test ran).
- `failure_output` — non-null on `FAILED`; excerpt of the test or lint
  output (bounded — do not dump multi-MB logs, ~40 lines is enough for
  a human to see the failure).

### `reason` values

On `verdict=PASSED`:

- `ok` — test passed with the fix applied; lint clean.

On `verdict=FAILED`:

- `test_still_failing` — Step 4 reported `FAILED`; fix is incomplete.
- `stale_skip_after_fix` — Step 4 reported `all skipped` with an XPU
  marker still in place.
- `xfail_after_fix` — Step 4 reported `xfailed`; fix did not turn the
  test green.
- `lint_errors_after_autofix` — Step 5 could not clean the lint after
  `spin fixlint`; a human touch is needed.
- `unrelated_files_touched_by_lint` — Step 5's `spin fixlint` modified
  files outside `changed_files`.

On `verdict=CANNOT_VERIFY`:

- `wheel_install_not_source` — Step 1 saw `torch` imported from
  `site-packages`; can't verify a source-tree fix through a wheel.
- `rebuild_failed` — `xpu-build-pytorch` returned failure during the
  Step 3 rebuild.
- `environmental_skip` — Step 4's `all skipped` had an environmental
  reason (missing dep, no accelerator).
- `test_collect_zero` — Step 4's `refined_command` resolves to zero
  collected tests; the fix cannot be validated against the reported
  reproducer.
- `test_timeout` — the test process exceeded its timeout and was
  killed.
- `other` — fallback; put full explanation in `reason_detail`.

The orchestrator decides whether to loop back to `fix-implement` on
`FAILED`, or proceed to commit/PR on `PASSED`, based on `verdict` and
`reason`.
