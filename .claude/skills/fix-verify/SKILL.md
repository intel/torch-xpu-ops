---
name: fix-verify
description: >
  Use when asked to verify a fix works, confirm a staged patch resolves a
  failure, or produce a before/after comparison of a fix. Runs the test
  command against a source build with the fix applied and reports
  PASSED / FAILED / CANNOT_VERIFY. Called by both issue-handler and
  xpu-nightly-ci-fix orchestrators after fix-implement.
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
- Step 3: [Before/after comparison](#step-3-beforeafter-comparison)
- Step 4: [Run test](#step-4-run-test)
- Step 5: [Lint](#step-5-lint)
- [Output](#output)

## Inputs

- `refined_command` — the exact test command from `fix-reproduce`'s
  output.
- `PYTORCH_DIR` — path to local PyTorch checkout.
- `target_repo_dir` — path to the checkout that holds the staged fix
  (same derivation rule as in `fix-implement`: equals `PYTORCH_DIR` for
  `target_repo=pytorch`, `<PYTORCH_DIR>/third_party/torch-xpu-ops` for
  `target_repo=torch-xpu-ops`). All `git stash`/`git diff --cached`
  operations run against `target_repo_dir`. The rebuild (Step 2) still
  runs from `PYTORCH_DIR` because `pip install -e .` builds pytorch and
  pulls its submodule pin.
- `changed_files` — list of changed files from `fix-implement`'s
  output; if any are C++/SYCL (`.cpp`, `.h`, `.cu`, `.sycl`), a rebuild
  is required before running.

This skill always runs the before/after comparison (Step 3) and lints
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

If any of `changed_files` are C++/SYCL (`.cpp`, `.h`, `.cu`, `.sycl`),
a rebuild is required.

**When `target_repo_dir != PYTORCH_DIR`** (i.e. the fix is inside
`third_party/torch-xpu-ops`), the pytorch build reads
`third_party/xpu.txt` and would clobber the staged fix by resetting
the submodule to the pinned commit. Apply the "Commit Pin & Development
Override" procedure from AGENTS.md **before** loading `xpu-build-pytorch`:

```bash
# Rewrite the pin to the working branch's HEAD so CMake's checkout
# becomes a no-op. Do NOT stage or commit this file (never staging
# third_party/xpu.txt is a HARD RULE in fix-implement).
git -C $target_repo_dir rev-parse HEAD > $PYTORCH_DIR/third_party/xpu.txt
```

**When the changed files include C++/SYCL sources**, DO NOT rebuild
first. A rebuild with the fix staged compiles the fix into
`torch/lib/*.so`, and the later "before" run (Step 3, after
`git stash -u`) would still load the fix's `.so` even with sources
removed — producing a false-negative "before" that already passes.
Instead, defer the rebuild into Step 3's before/after loop where it is
done at each phase.

If all changed files are python-only, no rebuild is needed here (the
before/after loop in Step 3 re-imports without a build step).

## Step 3: Before/after comparison

**Contract:** this step requires that `fix-implement` left changes
staged but uncommitted. `git stash -u` temporarily removes them to
obtain a before-state. If the stash
finds nothing (orchestrator committed the changes early), the contract
is violated — return
`CANNOT_VERIFY(reason=no_staged_changes)` (see below), do NOT silently
produce an after-only table.

All git commands here run against `target_repo_dir` (not `PYTORCH_DIR`);
these can differ when `target_repo == "torch-xpu-ops"`.

**When any `changed_files` are C++/SYCL, the before/after phases each
run their own rebuild** (see Step 2 rationale). For python-only changes
the rebuild lines below are no-ops.

```bash
# Capture the staged diff BEFORE stashing so we can compare after pop.
staged_before=$(git -C $target_repo_dir diff --cached)
[ -z "$staged_before" ] && \
  abort "no staged changes; fix-implement contract requires uncommitted staged changes"

# Record BEFORE (without the fix)
git -C $target_repo_dir stash -u   # stash staged, unstaged, untracked
# For torch-xpu-ops fixes, also restore xpu.txt to the base commit's pin
# so the rebuild sees the ORIGINAL submodule state.
if [ "$target_repo_dir" != "$PYTORCH_DIR" ]; then
    git -C $PYTORCH_DIR checkout -- third_party/xpu.txt
fi
# Rebuild WITHOUT the fix (only if C++/SYCL changed):
#   invoke xpu-build-pytorch skill here
# run test, capture output as before_output

# Restore the fix. `--index` restores the staged state the orchestrator
# commits from — but git will silently fall back to a non-index pop
# if the working tree conflicts with the popped state (e.g. a rebuild
# artifact appearing at a path that clashes with a stashed file).
git -C $target_repo_dir stash pop --index || \
  abort "git stash pop failed; staged fix state cannot be restored"

# Verify the staged diff matches what we captured before stashing.
# Compare via hash to handle diffs that exceed ARG_MAX.
before_sha=$(printf %s "$staged_before" | sha256sum | cut -d' ' -f1)
after_sha=$(printf %s "$(git -C $target_repo_dir diff --cached)" \
            | sha256sum | cut -d' ' -f1)
if [ "$before_sha" != "$after_sha" ]; then
    abort "git stash pop --index did not restore staged state"
fi

# Re-apply the xpu.txt override so the rebuild sees the working branch.
if [ "$target_repo_dir" != "$PYTORCH_DIR" ]; then
    git -C $target_repo_dir rev-parse HEAD > $PYTORCH_DIR/third_party/xpu.txt
fi
# Rebuild WITH the fix (only if C++/SYCL changed):
#   invoke xpu-build-pytorch skill here
# Record AFTER (with the fix)
# run test, capture output as after_output
```

The before/after outputs are folded into the markdown report (see
Output section) as a comparison table.

If `git stash -u` reports "No local changes to save", return
`CANNOT_VERIFY(reason=no_staged_changes)` with `blocker="no staged
changes; fix-implement contract requires uncommitted changes"`.

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
- `before_after_table` — the markdown table from Step 3 when both
  phases ran successfully; `null` only when Step 3 could not produce it
  (e.g. a `CANNOT_VERIFY` before reaching the after phase).
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
- `rebuild_failed` — `xpu-build-pytorch` returned failure in Step 2 or
  Step 3.
- `no_staged_changes` — Step 3 found nothing to stash; the contract
  from `fix-implement` (staged, uncommitted) is violated.
- `stash_pop_failed` — `git stash pop --index` failed or did not
  restore the staged state in Step 3.
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
