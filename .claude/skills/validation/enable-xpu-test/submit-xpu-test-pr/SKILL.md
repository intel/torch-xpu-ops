---
name: submit-xpu-test-pr
description: Submit a draft GitHub pull request for XPU test enablement after develop-xpu-test made the edits and verify-xpu-test confirmed them locally. Use when the XPU-enable changes (extended instantiate_device_type_tests to include "xpu"/HAS_GPU guard, and in-scope op_db DecorateInfo device_type widenings to ('cuda', 'xpu') for the target class in common_methods_invocations.py) are complete and verified, and you now need to rebase onto upstream pytorch/pytorch strict/viable, stage, commit, push to the daisyden/pytorch fork, and open a draft PR against pytorch/pytorch. Always rebases onto pytorch/pytorch viable/strict (or main) so the PR diff is only the enable change. Confirm-gated: never rebases, pushes, or creates a PR without explicit user approval. Regular fork PR via gh (NOT ghstack).
---

# Submit XPU Test PR

Submit a draft pull request for XPU test enablement work. This skill is the
**final** step of the enable-xpu-test pipeline:

```
develop-xpu-test  ->  verify-xpu-test  ->  submit-xpu-test-pr  (this skill)
```

It assumes the edits are already made (by `develop-xpu-test`) and already
locally XPU-verified (by `verify-xpu-test`). Its job is to rebase those edits
onto the upstream `pytorch/pytorch` strict/viable mainline and turn them into a
clean, confirm-gated draft PR against `pytorch/pytorch` (head = the
`daisyden/pytorch` fork branch).

The XPU-enable edits live in the **pytorch/pytorch checkout** (default
`$HOME/daisy_pytorch`), not in torch-xpu-ops:
- `test/<file>.py` — instantiation extended to include XPU.
- `torch/testing/_internal/common_methods_invocations.py` — `DecorateInfo`
  `device_type` entries widened to `('cuda', 'xpu')`.

## When to Use

- After `develop-xpu-test` + `verify-xpu-test` have completed and the changes
  are ready to review.
- When the user asks to open / submit / push a PR for XPU test enablement.

Do **not** use this skill to make code edits or to run the local verification —
those belong to `develop-xpu-test` and `verify-xpu-test`. This skill only stages,
commits, pushes, and opens the PR.

## Preconditions

1. **Verified changes present.** `verify-xpu-test` returned a **verified**
   verdict (no unreverted widened `expectedFailure` that unexpectedly passes on
   XPU). If verification was not run or did not pass, STOP and ask the user to
   run `verify-xpu-test` first.
2. **GitHub auth.** `gh auth status` succeeds with a token that can push to the
   `daisyden/pytorch` fork and open PRs against `pytorch/pytorch`.
3. **Git remotes** in the pytorch checkout:
   - a fork remote (`origin` or `daisyden`) ->
     `https://github.com/daisyden/pytorch.git`
   - `upstream` -> `https://github.com/pytorch/pytorch.git` (REQUIRED — the PR
     base and rebase target; Step 2.5 adds it if missing)

   Verify and, if missing, ask the user before adding a remote.

## Tools Used

- **bash**: `git`, `gh` (status, diff, commit, push, pr create).
- **read / grep**: inspect the diff before drafting the commit/PR message.
- **question**: get explicit user approval before push and before PR creation.

## Workflow

Run all git/gh commands from the pytorch checkout directory
(`<pytorch_folder>`, default `$HOME/daisy_pytorch`).

### Step 1: Inspect the Working Tree

Confirm the changes are exactly the XPU-enable edits and nothing stray.

```bash
cd <pytorch_folder>
git status
git diff --stat
git diff
```

Sanity checks on the diff:
- Only expected files changed: the enabled `test/<file>.py` and/or
  `torch/testing/_internal/common_methods_invocations.py`.
- The diff contains XPU-enable tokens (`only_for=("cuda", "xpu")`,
  `allow_xpu=True`, `device_type=('cuda', 'xpu')`, or the `HAS_GPU` guard).
- The diff does **not** add new `@skipIfXpu` / `skipXPU` skip decorators and
  does **not** modify pre-existing XPU skips/decorators (those are out of scope
  for enablement — flag to the user if present).
- **Every changed `DecorateInfo` is in scope.** Each modified
  `common_methods_invocations.py` entry must reference the target test class and
  one of the generic test names that class actually runs. A changed
  `DecorateInfo` for a different class (`TestForeach`, `TestUnaryUfuncs`,
  `TestInductorOpInfo`, `TestSparseCSR`, ...) or an unrelated test name is
  out of scope: STOP and have the caller re-run `develop-xpu-test` /
  `verify-xpu-test` to re-scope. If the target class has no matching op_db
  entries, `common_methods_invocations.py` must be unchanged.

If unexpected files are dirty, ask the user how to proceed (stash / exclude /
abort). Do not blindly `git add -A`. Ignore untracked dev-only paths such as
`third_party/torch-xpu-ops/` (never stage them).

### Step 2: Choose / Create the Branch

```bash
git branch --show-current
```

- If already on a dedicated feature branch, use it.
- Otherwise create one:
  ```bash
  git checkout -b xpu/enable-<short-scope>   # e.g. xpu/enable-test-sdpa
  ```

Never commit XPU-enable work directly onto `main`/`master`.

### Step 2.5: Rebase onto upstream pytorch/pytorch (MANDATORY)

The PR base MUST be the upstream `pytorch/pytorch` mainline, and the branch MUST
be rebased onto it. This is required because the local dev checkout is usually
synced to the installed torch **wheel commit**, which is frequently on a
**release branch** (e.g. `release/2.13`), not `main`. Opening a PR from such a
branch against `main` produces a diff full of unrelated release-vs-main files.

Do this before committing/pushing:

```bash
# Ensure an upstream remote for pytorch/pytorch exists.
git remote -v | grep -q 'pytorch/pytorch' || \
  git remote add upstream https://github.com/pytorch/pytorch.git

# Fetch the strict/viable mainline. Prefer the known-good viable/strict branch;
# fall back to main if viable/strict is not available.
git fetch upstream viable/strict 2>/dev/null && UPSTREAM_BASE=upstream/viable/strict || {
  git fetch upstream main && UPSTREAM_BASE=upstream/main
}
echo "Rebasing onto ${UPSTREAM_BASE}"

# Confirm the target file's changed line still exists on the upstream base;
# rebase (or cherry-pick the enable commit) onto it.
git rebase "${UPSTREAM_BASE}"
```

Notes:
- **`viable/strict`** is PyTorch's continuously-green snapshot of `main`; it is
  the preferred base. If it is unavailable, use `main`.
- The PR is opened against the **branch name** (`viable/strict` or `main`) on
  `pytorch/pytorch`; GitHub resolves it in the upstream repo. Never open the PR
  against a release branch or a fork-local snapshot branch.
- If the enable edits were made on a release-synced checkout, the clean way to
  re-base a single-commit change is:
  ```bash
  git checkout -b xpu/enable-<short-scope>-main "${UPSTREAM_BASE}"
  git cherry-pick <enable-commit>     # resolve the 1-file change onto mainline
  ```
- If the rebase/cherry-pick conflicts, resolve using the mainline version of the
  file (the enable change is a small, mechanical edit that should re-apply
  cleanly). If it cannot re-apply cleanly, STOP and report — the file may have
  changed on mainline in a way that needs re-verification.
- After rebasing, re-confirm the diff vs the upstream base is still exactly the
  in-scope enable change:
  ```bash
  git diff --stat "${UPSTREAM_BASE}"..HEAD   # expect only the target file(s)
  ```

### Step 3: Draft the Commit

Stage only the intended files (explicit paths, never `-A` blindly):

```bash
git add test/<file>.py torch/testing/_internal/common_methods_invocations.py
```

Draft a commit message following the repo convention (imperative subject, a
body explaining the "why", and a Test Plan section with the literal verification
commands). Example:

```
[XPU][Test] Enable <ClassName> / <op(s)> on XPU

Extend instantiate_device_type_tests to only_for=("cuda", "xpu") with
allow_xpu=True, and widen the CUDA-registered DecorateInfo xfail/skip entries
in common_methods_invocations.py to device_type=('cuda', 'xpu') so the same
expected failures apply on XPU. No new XPU skip decorators are added and no
existing XPU skips/decorators are changed.

Test Plan:
  source ~/miniforge3/bin/activate pytorch_opencode_env
  cd /tmp
  python -m pytest <repo>/test/<file>.py -v -k "<ClassName> and xpu" --tb=short
  python -m pytest <repo>/test/test_ops.py -v -k "<op_name> and xpu" --tb=short

Authored with an AI assistant.
```

Notes:
- Disclose AI assistance in the body ("Authored with an AI assistant.").
- Do **NOT** add a `Co-authored-by:` AI trailer (interferes with the Linux
  Foundation CLA bot).
- Do not commit yet — present the message in Step 4 first.

### Step 4: Confirm Before Committing / Pushing (MANDATORY)

Present to the user, via the `question` tool:
- the file list and `git diff --stat` **against the upstream base** (Step 2.5),
- the drafted commit message,
- the target: branch `<branch>` (head `daisyden:<branch>`) -> upstream
  `pytorch/pytorch`, base `viable/strict` (or `main`), draft PR.

Ask for explicit approval (approve / edit / abort). Only on **approve** proceed.

### Step 5: Commit and Push to the Fork

```bash
git commit -m "$(cat <<'EOF'
<approved commit message>
EOF
)"

# Push to the daisyden/pytorch fork. Use --force-with-lease (never --force),
# and never force-push main/master.
git push -u origin <branch> --force-with-lease
```

Force-push rules: always `--force-with-lease`, never plain `--force`; if it
rejects, fetch and reconcile — do not escalate to `--force`.

### Step 6: Open the Draft PR

Open a **draft** PR against the upstream `pytorch/pytorch` mainline, using `gh`.
The base is the same strict/viable branch you rebased onto in Step 2.5
(`viable/strict`, or `main` if `viable/strict` is unavailable). This is a
regular fork PR (head = `daisyden:<branch>`) — **NOT ghstack**.

```bash
# BASE_BRANCH is the branch name on pytorch/pytorch you rebased onto:
#   viable/strict  (preferred)  or  main
BASE_BRANCH=main   # set to viable/strict when that was the rebase base

gh pr create \
  --repo pytorch/pytorch \
  --base "${BASE_BRANCH}" \
  --head daisyden:<branch> \
  --draft \
  --title "[XPU][Test] Enable <ClassName> / <op(s)> on XPU" \
  --body "$(cat <<'EOF'
## Summary
- Extend `instantiate_device_type_tests` for `<ClassName>` to
  `only_for=("cuda", "xpu"), allow_xpu=True` (or add a `HAS_GPU` guard).
- Widen the CUDA-registered `DecorateInfo` xfail/skip/tolerance entries that
  belong to `<ClassName>` in
  `torch/testing/_internal/common_methods_invocations.py` to
  `device_type=('cuda', 'xpu')` so the same expected failures apply on XPU.
  (Only entries for `<ClassName>`'s tests are touched; unrelated classes are
  left unchanged. If none matched, op_db is unchanged.)
- No new XPU skip decorators added; existing XPU skips/decorators untouched.

## Test plan
- XPU: enabled variants run and behave as expected (pass / skip / xfail):
  ```
  python -m pytest <repo>/test/<file>.py -v -k "<ClassName> and xpu" --tb=short
  python -m pytest <repo>/test/test_ops.py -v -k "<op_name> and xpu" --tb=short
  ```
- CUDA behavior unchanged (validated by CI on a CUDA host).

Authored with an AI assistant.
EOF
)"
```

After creation, verify the PR base and diff are correct:

```bash
gh pr view <num> --repo pytorch/pytorch --json baseRefName,files,isDraft,url
# Expect baseRefName == main (or viable/strict) and files == only the target file(s).
```

Return the PR URL to the user.

### Step 7: Report

Report: branch name, commit hash, fork push result, and the draft PR URL.

## Constraints

1. **Confirm-gated.** NEVER `git commit`, `git push`, or `gh pr create` without
   explicit user approval (draft -> confirm -> submit). This applies even when
   invoked automatically by an upstream orchestrator.
2. **Base is upstream pytorch/pytorch strict/viable.** The PR MUST be opened
   against `pytorch/pytorch` on the `viable/strict` branch (preferred) or `main`,
   and the branch MUST be rebased onto that base first (Step 2.5). NEVER open the
   PR against a release branch, a wheel-commit snapshot, or a fork-local base
   branch. Head is the fork branch (`daisyden:<branch>`); the PR is a draft.
3. **Regular fork PR, NEVER ghstack.** Do not use ghstack for this workflow.
4. **Force-push safety.** Always `--force-with-lease`, never `--force`; never
   force-push to `main`/`master`.
5. **Stage explicit paths.** Never `git add -A` / `git add .` blindly — add only
   the intended test file(s) and `common_methods_invocations.py`. Never stage
   `third_party/torch-xpu-ops/` or other untracked dev paths.
6. **No new edits, in-scope only.** This skill does not modify test code, op_db,
   or run local verification — it only packages the already-verified changes into
   a PR. If the diff contains new XPU skip decorators, touches existing XPU
   skips/decorators, or changes any `DecorateInfo` for a test class/name outside
   the target class, flag it and stop (out of scope for enablement).
7. **AI disclosure, no CLA-breaking trailer.** Disclose AI assistance in the
   commit/PR body; do NOT add a `Co-authored-by:` AI trailer.
8. **ASCII only** in authored commit/PR content.

## See Also

- `develop-xpu-test` — makes the XPU-enable edits this skill submits.
- `verify-xpu-test` — locally verifies the edits before this skill packages them.
