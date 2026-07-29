---
name: submit-refactoring-pr
description: >
  Submit a GitHub pull request for a PyTorch test refactoring (S1/S2/S3
  decoupling) after the refactor-test-decoupling edits are done and verified.
  Use when the refactoring split is complete, test count has been verified, and
  you now need to rebase the changes onto upstream pytorch/pytorch
  strict/viable, stage, commit, push to the daisyden/pytorch fork, and open a
  draft PR against pytorch/pytorch. The PR diff must be only the refactored test
  file(s), and the commit/PR body must include evidence that no test cases were
  lost.
---

# Submit Refactoring PR

Submit a draft pull request for PyTorch test refactoring work. This skill is the
**final** step of the refactoring pipeline:

```
refactor-test-decoupling  ->  review-test-refactoring  ->  submit-refactoring-pr  (this skill)
```

It assumes the refactoring edits are already made (by `refactor-test-decoupling`)
and already reviewed (by `review-test-refactoring`). Its job is to rebase those
edits onto the upstream `pytorch/pytorch` strict/viable mainline and turn them
into a clean, confirm-gated draft PR against `pytorch/pytorch` (head = the
`daisyden/pytorch` fork branch).

## When to Use

- After `refactor-test-decoupling` + `review-test-refactoring` have completed
  and the refactored test file(s) are ready to review.
- When the user asks to open / submit / push a PR for a test decoupling
  refactoring.
- When a test file has been split into S1 (accelerator-unrelated), S2
  (accelerator-agnostic), and/or S3 (accelerator-specific) classes, and the
  split is verified.

Do **not** use this skill to make code edits or to run the review — those
belong to `refactor-test-decoupling` and `review-test-refactoring`. This skill
only stages, commits, pushes, and opens the PR.

## Preconditions

1. **Verified refactoring.** `review-test-refactoring` (or equivalent manual
   review) has confirmed:
   - Test count is preserved (original == refactored).
   - Naming conventions are correct (S1=`TestFoo`, S2=`TestFooDevice`,
     S3=`TestFooCUDA`/`TestFooXPU`/etc).
   - Classification hierarchy is correct (S3 > S2 > S1).
   - No test logic was changed — only split/moved.
   - Syntax compiles cleanly (`py_compile`).
   - No external references (op_db, dynamo_skips, etc.) need updating for
     the renamed classes.
2. **GitHub auth.** `gh auth status` succeeds with a token that can push to the
   `daisyden/pytorch` fork and open PRs against `pytorch/pytorch`.
3. **Git remotes** in the pytorch checkout:
   - a fork remote (`origin` or `daisyden`) ->
     `https://github.com/daisyden/pytorch.git`
   - `upstream` -> `https://github.com/pytorch/pytorch.git` (REQUIRED — the PR
     base and rebase target; the workflow adds it if missing)

   Verify and, if missing, ask the user before adding a remote.

## Tools Used

- **bash**: `git`, `gh` (status, diff, commit, push, pr create).
- **read / grep**: inspect the diff and collect test count evidence.
- **question**: get explicit user approval before push and before PR creation.

## Workflow

Run all git/gh commands from the pytorch checkout directory (`<pytorch_folder>`).

### Step 1: Inspect the Working Tree

Confirm the changes are exactly the refactoring edits and nothing stray.

```bash
cd <pytorch_folder>
git status
git diff --stat
git diff
```

Sanity checks on the diff:
- Only expected test files changed (refactored `test/<file>.py` file(s)).
- The diff contains only:
  - Test method movements between classes (S1 ↔ S2 ↔ S3).
  - Class renames (e.g. `TestFooDeviceType` → `TestFooDevice`).
  - New S2/S3 class definitions with `instantiate_device_type_tests`.
  - Import additions for decoupling (e.g. `onlyAccelerator`,
    `instantiate_device_type_tests`).
  - Removal of stale CUDA guards that were replaced with accelerator-agnostic
    alternatives.
- The diff does **not** contain:
  - Changes to test logic or assertions.
  - Unrelated files or formatting-only changes.

If unexpected files are dirty, ask the user how to proceed (stash / exclude /
abort). Do not blindly `git add -A`. Ignore untracked dev-only paths such as
`third_party/torch-xpu-ops/` (never stage them).

### Step 2: Collect Test Count Evidence

Run these to produce the evidence you'll include in the commit/PR message:

```bash
# Count test methods in each changed test file
for f in test/<file1>.py test/<file2>.py; do
  echo "$f: $(grep -c 'def test_' "$f") test methods"
done

# Compare with original (check stash or use git show HEAD)
git stash
for f in test/<file1>.py test/<file2>.py; do
  echo "Original $f: $(grep -c 'def test_' "$f") test methods"
done
git stash pop
```

Alternative (no stash needed, compare against the base after rebase):

```bash
# After rebasing in Step 3:
git diff --stat <base>..HEAD       # files changed
git diff <base>..HEAD -- test/<file>.py | grep '^[+-]' | grep -c 'def test_'
```

### Step 3: Choose / Create the Branch

```bash
git branch --show-current
```

- If already on a dedicated feature branch (detached HEAD in a dev checkout),
  create one:
  ```bash
  git checkout -b refactor/<short-scope>   # e.g. refactor/test_dataloader
  ```

Never commit refactoring work directly onto `main`/`master`.

### Step 4: Rebase onto upstream pytorch/pytorch (MANDATORY)

The PR base MUST be the upstream `pytorch/pytorch` mainline, and the branch MUST
be rebased onto it. This is required because the local dev checkout is usually
synced to the installed torch **wheel commit**, which is frequently on a
**release-version snapshot** (e.g. a nightly), not `main`. Opening a PR from
such a branch against `main` produces a diff full of unrelated release-vs-main
files.

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

# Rebase the branch onto the upstream base.
git rebase "${UPSTREAM_BASE}"
```

Notes:
- **`viable/strict`** is PyTorch's continuously-green snapshot of `main`; it is
  the preferred base. If it is unavailable, use `main`.
- The PR is opened against the **branch name** (`viable/strict` or `main`) on
  `pytorch/pytorch`; GitHub resolves it in the upstream repo.
- If the refactoring edits were made on a release-synced checkout, use:
  ```bash
  git checkout -b refactor/<short-scope> "${UPSTREAM_BASE}"
  # Re-apply the changes (cherry-pick or manual patch)
  ```
- If the rebase conflicts, resolve using the mainline version of the file (the
  refactoring diff is a mechanical test split that should re-apply cleanly).
  If it cannot re-apply cleanly, STOP and report — the file may have changed on
  mainline in a way that needs re-verification.
- After rebasing, re-confirm the diff vs the upstream base is still exactly the
  in-scope refactoring change:
  ```bash
  git diff --stat "${UPSTREAM_BASE}"..HEAD
  ```

### Step 5: Draft the Commit

Stage only the intended files (explicit paths, never `-A` blindly):

```bash
git add test/<file1>.py test/<file2>.py
```

Draft a commit message following the repo convention (imperative subject, a
body explaining the "why", and a Test Plan section with the verification
evidence).

Commit message structure:

```
[Test][Refactor] Decouple <ClassName> tests according to the accelerator relevance.

Summary of changes:
- <ClassName>: N tests → N tests (unchanged)
  - Moved CUDA-specific tests to new <FooCUDA> class (S3)
  - Moved accelerator-agnostic device-type tests to new <FooDevice> class (S2)
  - Remaining <N> tests stay in <Foo> (S1, accelerator-unrelated)
  - Renamed <FooDeviceType> → <FooDevice> (S2 naming convention)

Test count preserved:
  test/<file>.py: <original> test methods → <refactored> test methods ✓

Evidence:
  $ grep -c 'def test_' test/<file>.py
  <count>

Classification logic (per refactor-test-decoupling skill):
  - S1 (accelerator-unrelated): tests that use no device APIs
  - S2 (accelerator-agnostic): tests that use device-generic APIs
    (torch.accelerator, instantiate_device_type_tests)
  - S3 (accelerator-specific): tests that use CUDA-only APIs
    (torch.cuda, CUDA IPC, NCCL, etc.)

Authored with an AI assistant.
```

Do not commit yet — present the message in Step 6 first.

### Step 6: Confirm Before Committing / Pushing (MANDATORY)

Present to the user, via the `question` tool:
- the file list and `git diff --stat` **against the upstream base** (Step 4),
- the test count evidence (original vs refactored),
- the drafted commit message,
- the target: branch `<branch>` (head `daisyden:<branch>`) -> upstream
  `pytorch/pytorch`, base `viable/strict` (or `main`), draft PR.

Ask for explicit approval (approve / edit / abort). Only on **approve** proceed.

### Step 7: Commit and Push to the Fork

```bash
git commit -m "$(cat <<'EOF'
<approved commit message>
EOF
)"

# Push to the daisyden/pytorch fork. Use --force-with-lease (never --force),
# and never force-push main/master.
git push -u <fork_remote> <branch> --force-with-lease
```

Force-push rules: always `--force-with-lease`, never plain `--force`; if it
rejects, fetch and reconcile — do not escalate to `--force`.

### Step 8: Open the Draft PR

Open a **draft** PR against the upstream `pytorch/pytorch` mainline, using `gh`.
The base is the same strict/viable branch you rebased onto in Step 4
(`viable/strict`, or `main` if `viable/strict` is unavailable). This is a
regular fork PR (head = `daisyden:<branch>`) — **NOT ghstack**.

```bash
# BASE_BRANCH is the branch name on pytorch/pytorch you rebased onto:
#   viable/strict  (preferred)  or  main
BASE_BRANCH=viable/strict   # set to main when that was the rebase base

gh pr create \
  --repo pytorch/pytorch \
  --base "${BASE_BRANCH}" \
  --head daisyden:<branch> \
  --draft \
  --title "[Test][Refactor] Decouple <ClassName> tests according to the accelerator relevance." \
  --body "$(cat <<'EOF'
## Summary
- Refactored `test/<file>.py` following the S1/S2/S3 decoupling strategy:
  - **S1 (accelerator-unrelated)**: `<Foo>` — tests that use no device APIs.
  - **S2 (accelerator-agnostic)**: `<FooDevice>` — tests using
    `instantiate_device_type_tests` with `onlyAccelerator` or
    accelerator-generic device guards.
  - **S3 (accelerator-specific)**: `<FooCUDA>` — tests using CUDA-only APIs
    (CUDA IPC, `torch.cuda`, NCCL, etc.).
- Renamed `<FooDeviceType>` → `<FooDevice>` per S2 naming convention.
- No test logic changed; only split/moved between classes.

## Test evidence
Test count preserved:
- Original: `<count>` test methods
- Refactored: `<count>` test methods

```
$ grep -c 'def test_' test/<file>.py
<count>
```

## Refactoring strategy
Per the refactor-test-decoupling methodology:
- S3 > S2 > S1 classification hierarchy strictly followed.
- Whitelist decorators (`@onlyCUDA`, `@unittest.skipIf(not TEST_CUDA, ...)`)
  enlarged to `@onlyAccelerator`.
- Blacklist decorators (`@skipIfXpu`, `@skipIfRocm`, `@onlyNativeDeviceTypesAnd`)
  preserved as-is.
- Category C APIs (NCCL, NVTX, cuDNN) are the only patterns that make a test S3.
- `instantiate_device_type_tests` used for S2 classes.

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

### Step 9: Report

Report: branch name, commit hash, fork push result, and the draft PR URL.

## Constraints

1. **Confirm-gated.** NEVER `git commit`, `git push`, or `gh pr create` without
   explicit user approval (draft -> confirm -> submit). This applies even when
   invoked automatically by an upstream orchestrator.
2. **Base is upstream pytorch/pytorch strict/viable.** The PR MUST be opened
   against `pytorch/pytorch` on the `viable/strict` branch (preferred) or `main`,
   and the branch MUST be rebased onto that base first (Step 4). NEVER open the
   PR against a release branch, a wheel-commit snapshot, or a fork-local base
   branch. Head is the fork branch (`daisyden:<branch>`); the PR is a draft.
3. **Regular fork PR, NEVER ghstack.** Do not use ghstack for this workflow.
4. **Force-push safety.** Always `--force-with-lease`, never `--force`; never
   force-push to `main`/`master`.
5. **Stage explicit paths.** Never `git add -A` / `git add .` blindly — add only
   the intended test file(s). Never stage `third_party/torch-xpu-ops/` or other
   untracked dev paths.
6. **No logic changes.** This skill does not modify test logic, add/remove test
   methods, or change assertions — it only packages the already-verified
   refactoring changes into a PR. If the diff contains non-refactoring changes
   (logic edits, unrelated files), flag them and stop.
7. **Test count evidence required.** Every commit/PR MUST include evidence that
   the total number of test methods is preserved before and after the refactoring.
   Include the `grep -c 'def test_'` output in the commit body and PR description.
8. **AI disclosure, no CLA-breaking trailer.** Disclose AI assistance in the
   commit/PR body; do NOT add a `Co-authored-by:` AI trailer.
9. **ASCII only** in authored commit/PR content.

## See Also

- `refactor-test-decoupling` — makes the S1/S2/S3 refactoring edits this skill
  submits.
- `review-test-refactoring` — reviews the refactoring for correctness before
  this skill packages it.
