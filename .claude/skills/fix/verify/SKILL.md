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
- `target_repo_dir` — path to the checkout that holds the staged fix
  (same derivation rule as in `fix/implement`: equals `pytorch_dir` for
  `target_repo=pytorch`, `<pytorch_dir>/third_party/torch-xpu-ops` for
  `target_repo=torch-xpu-ops`). All `git stash`/`git diff --cached`
  operations run against `target_repo_dir`. The rebuild (Step 2) still
  runs from `pytorch_dir` because `pip install -e .` builds pytorch
  and pulls its submodule pin.
- `changed_files` — list of changed files; if any are C++/SYCL
  (`.cpp`, `.h`, `.cu`, `.sycl`), a rebuild is required before running.
- `run_before_after_diff` (bool, default `false`) — if `true`, runs the test
  before and after the fix to produce a comparison table. Set to `true` by
  `nightly-ci-fix`.
- `run_lint` (bool, default `false`) — if `true`, runs `spin fixlint` after a
  passing result. Set to `true` by `nightly-ci-fix`.

### Shell helpers

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

The installed `torch` must be the one built from `pytorch_dir`, not a
wheel. Checking `torch.version.git_version` is NOT sufficient —
released and nightly wheels also carry a real commit hash there.
Check where `torch` is imported from instead:

```bash
python -c "import torch, os; print(os.path.realpath(torch.__file__))"
```

The printed path must be under `$(realpath <pytorch_dir>)/torch/`. If it
resolves into `site-packages` of an unrelated prefix, the environment is
a wheel install: return `CANNOT_VERIFY` with `blocker: "wheel install
(torch imported from <path>); verify requires a source build"` — a fix
staged in the local tree has no effect on an installed wheel. Producing
the source build is the orchestrator's job (both orchestrators load
`xpu-build-pytorch` before calling this skill when `fix/reproduce`
reproduced at `stage=nightly`).

## Step 2: Rebuild if needed

If any of `changed_files` are C++/SYCL (`.cpp`, `.h`, `.cu`, `.sycl`),
a rebuild is required.

**When `target_repo_dir != pytorch_dir`** (i.e. the fix is inside
`third_party/torch-xpu-ops`), the pytorch build reads
`third_party/xpu.txt` and would clobber the staged fix by resetting
the submodule to the pinned commit. Apply the "Commit Pin & Development
Override" procedure from AGENTS.md **before** loading `xpu-build-pytorch`:

```bash
# Rewrite the pin to the working branch's HEAD so CMake's checkout
# becomes a no-op. Do NOT stage or commit this file (never stage
# third_party/xpu.txt is a HARD RULE in fix/implement).
git -C <target_repo_dir> rev-parse HEAD > <pytorch_dir>/third_party/xpu.txt
```

**When `run_before_after_diff=true` and the changed files include
C++/SYCL sources**, DO NOT rebuild first. A rebuild with the fix
staged compiles the fix into `torch/lib/*.so`, and the later "before"
run (Step 3, after `git stash -u`) would still load the fix's `.so`
even with sources removed — producing a false-negative "before" that
already passes. Instead, defer the rebuild into Step 3's before/after
loop where it is done at each phase.

Otherwise (i.e. `run_before_after_diff=false`, or all changes are
python-only regardless of the flag):

- If any changed file is C++/SYCL: load `xpu-build-pytorch` and
  rebuild now.
- If all changed files are python-only: no rebuild needed.

## Step 3: Before/after comparison (if run_before_after_diff=true)

**Contract:** this step requires that `fix/implement` left changes staged but
uncommitted. `git stash -u` temporarily removes them to obtain a before-state.
If `run_before_after_diff=true` and the stash finds nothing (orchestrator
committed the changes early), the contract is violated — return `CANNOT_VERIFY`
(see below), do not silently produce an after-only table.

All git commands here run against `target_repo_dir` (not `pytorch_dir`);
these can differ when `target_repo == "torch-xpu-ops"`.

**When any `changed_files` are C++/SYCL, the before/after phases each
run their own rebuild** (see Step 2 rationale). For python-only
changes the rebuild lines below are no-ops.

```bash
# Capture the staged diff BEFORE stashing so we can compare after pop.
staged_before=$(git -C <target_repo_dir> diff --cached)
[ -z "$staged_before" ] && \
  abort "CANNOT_VERIFY: no staged changes; fix/implement contract requires uncommitted staged changes"

# Record BEFORE (without the fix)
git -C <target_repo_dir> stash -u   # stash staged, unstaged, untracked
# For torch-xpu-ops fixes, also restore xpu.txt to the base commit's pin
# so the rebuild sees the ORIGINAL submodule state.
if [ "<target_repo_dir>" != "<pytorch_dir>" ]; then
    git -C <pytorch_dir> checkout -- third_party/xpu.txt
fi
# Rebuild WITHOUT the fix (only if C++/SYCL changed):
#   invoke xpu-build-pytorch skill here
# run test, capture output as before_output

# Restore the fix. `--index` restores the staged state the orchestrator
# commits from — but git will silently fall back to a non-index pop
# if the working tree conflicts with the popped state (e.g. a rebuild
# artifact appearing at a path that clashes with a stashed file).
git -C <target_repo_dir> stash pop --index || \
  abort "CANNOT_VERIFY: git stash pop failed; staged fix state cannot be restored"

# Verify the staged diff matches what we captured before stashing.
# Compare via hash to handle diffs that exceed ARG_MAX.
before_sha=$(printf %s "$staged_before" | sha256sum | cut -d' ' -f1)
after_sha=$(printf %s "$(git -C <target_repo_dir> diff --cached)" \
            | sha256sum | cut -d' ' -f1)
if [ "$before_sha" != "$after_sha" ]; then
    abort "CANNOT_VERIFY: git stash pop --index did not restore staged state"
fi

# Re-apply the xpu.txt override so the rebuild sees the working branch.
if [ "<target_repo_dir>" != "<pytorch_dir>" ]; then
    git -C <target_repo_dir> rev-parse HEAD > <pytorch_dir>/third_party/xpu.txt
fi
# Rebuild WITH the fix (only if C++/SYCL changed):
#   invoke xpu-build-pytorch skill here
# Record AFTER (with the fix)
# run test, capture output as after_output
```

Output a comparison table:

```
| Test case | Before | After |
|-----------|--------|-------|
| TestFooXPU::test_bar | FAILED (AssertionError: ...) | PASSED |
```

If `git stash -u` reports "No local changes to save", return `CANNOT_VERIFY`
with `blocker: "no staged changes; fix/implement contract requires uncommitted changes"`.

## Step 4: Run test

Run ALL failing test cases from the original report individually.

Result interpretation:

- `all skipped` → read pytest's `SKIPPED [N] <file:line>: <reason>` line
  to classify:
  - Reason matches an XPU marker (`skipIfXpu`, `xfailIfXPU`,
    `expectedFailureXPU`, `skipXPU`, or a message containing "xpu")
    → `FAILED`. Rationale: a fix that leaves the failing test skipped
    is an incomplete fix. This is intentionally different from
    `fix/reproduce`'s handling - reproduce temporarily unskips to
    confirm the bug, but verify's job is to confirm the fix; if the
    XPU-marker skip is still in place, the fix did not touch what it
    should have. Include suggestion: "test still skipped after fix -
    the stale skip decorator should have been removed as part of the
    fix (see `fix/skip-management`)."
  - Reason is environmental (missing dep, no GPU, tool not found, "no
    accelerator", etc.) → `CANNOT_VERIFY` with
    `blocker: "test skipped for environmental reason: <marker>"`.
- `xfailed` → `FAILED`.

## Step 5: Lint (if run_lint=true)

Only run after a passing test result. Run it in `target_repo_dir` — the
repo that owns the changed files — not in `pytorch_dir`:

```bash
cd <target_repo_dir>
spin fixlint
# fixlint rewrites files in the working tree, which UNSTAGES those
# hunks. The orchestrator commits what is staged, so re-stage the
# same files or the lint fixes are silently dropped from the commit.
git -C <target_repo_dir> add -- <changed_files>
# Nothing outside changed_files may become staged; if `git status`
# shows fixlint touched other files, return FAILED rather than
# widening the diff.
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
