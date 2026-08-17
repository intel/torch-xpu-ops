---
name: fix/skip-list
description: >
  Process a "Bug Skip" tracking issue — a list of already-skipped tests
  asking whether each skip is still needed. For each entry: re-verify
  via fix/reproduce, classify (still-failing / already-fixed / env /
  flaky), and for still-failing entries run the full fix pipeline as a
  sub-bug producing one patch-proposal per test. Called by issue-handler
  Stage 1 when issue-triage returns issue_type=skip-list.
---

# Skip-List — Bug Skip Issue Handler

Mini-orchestrator for skip-list issues. Loops `fix/reproduce` over each
test entry, then recursively runs the full bug pipeline
(`fix/root-cause` → `fix/implement` → `fix/verify` → review) for
still-failing entries. Peer of `fix/root-cause` (not a child of it); the
top-level orchestrator dispatches to this skill instead of running
the bug pipeline directly when the issue is a skip-list.

## Inputs

- `issue_body` — full GitHub issue body containing the test-entry
  checklist. Entries marked `~~strike-through~~` are already-resolved
  and MUST be skipped.
- `pytorch_dir` — local pytorch checkout path.
- `pr_repo` — the ONE repo this run is allowed to open a PR against.
  In skip-list processing the deliverable is always patch-proposals on the
  issue, not PRs — so this skill treats every sub-bug as
  patch-proposal mode regardless of `pr_repo`. The orchestrator still
  passes `pr_repo` so future work can lift this restriction if
  needed.
- `pytorch_base` — commit/ref the pytorch checkout is reset to between
  sub-bugs (`origin/main`, or the CI commit sha if `fix/reproduce` fell
  back to it).
- `xpu_ops_base` — commit/ref the `third_party/torch-xpu-ops` override
  checkout is reset to between sub-bugs (that repo's own `origin/main`
  unless the caller pinned it). Separate from `pytorch_base` because a
  pytorch sha does not exist in torch-xpu-ops.

## Pipeline

```
parse entries → per-entry reproduce → classify → per-sub-bug pipeline
             → verdict table + per-sub-bug patch-proposal comments
```

### Step 1 — Parse entries

Extract the test node ids from the issue body's checklist. Entries
wrapped in `~~...~~` are already-resolved (a prior run or a human
struck them through); drop them from the working set. The remaining
entries are the input list for Step 2.

### Step 2 — Per-entry reproduce

For each remaining entry, call `fix/reproduce` with the entry as the
`reproducer_command` (this is the "per-entry loop" provider documented
in `fix/reproduce` Inputs).

### Step 3 — Classify each reproduce result

| reproduce output | classification |
|---|---|
| `REPRODUCED` (FAILED) | `STILL_FAILING` (repo owner unknown yet — decided by triage in Step 5) |
| `NOT_REPRODUCED` (PASSED on nightly / source build) | `ALREADY_FIXED` |
| `NO_REPRODUCER` (test does not exist / `collected 0 items`) or `CANNOT_VERIFY` (test-name drift) | `ENVIRONMENT` |
| Intermittent (pass on retry) | `FLAKY` |

`fix/reproduce` only reports whether the test fails — it does NOT
determine which repo owns the root cause. That decision is made by
the per-entry `fix/root-cause` run in Step 5. The verdict table (assembled
in Step 6, after Step 5 completes) records each `STILL_FAILING` entry
as either `STILL_FAILING_UPSTREAM_BUG` (root cause in pytorch) or
`STILL_FAILING_XPU_BUG` (root cause in torch-xpu-ops), using
`target_repo` from that entry's triage output.

### Step 4 — Verdict table (deferred to Step 6)

The per-test verdict table is a mandatory deliverable but is built
**after** Step 5 completes, because `STILL_FAILING` entries need their
`target_repo` (from triage) to be recorded as
`STILL_FAILING_UPSTREAM_BUG` or `STILL_FAILING_XPU_BUG`. Placeholder
here; final table format lives in Step 6.

### Step 5 — Per-sub-bug pipeline (STILL_FAILING only)

For every `STILL_FAILING` entry, treat that single test as a
**sub-bug** and run the full bug pipeline in patch-proposal mode.

**Isolation between sub-bugs.** Sub-bugs share the same checkouts,
but each sub-bug's `git diff --cached` MUST contain only that
sub-bug's changes (the HARD RULE "NEVER bundle multiple sub-bug
fixes into one diff" would otherwise be violated by construction —
`fix/implement` leaves changes staged, and without cleanup the next
sub-bug's `fix/implement` accumulates on top of the previous one).

**Two checkouts to consider.** Because different sub-bugs may triage
to different repos, both potential target checkouts must be reset
between sub-bugs, not just the last one used:

- `pytorch_dir` — used when `target_repo=pytorch`.
- `<pytorch_dir>/third_party/torch-xpu-ops` — used when
  `target_repo=torch-xpu-ops`.

**Base commits for reset.** Two separate values because the two repos
have distinct commit graphs — a pytorch sha does NOT exist in
torch-xpu-ops:

- `pytorch_base` — same `<base>` the top-level orchestrator would
  have branched from: `origin/main` in the normal case, or the CI
  commit sha if `fix/reproduce` fell back to it.
- `xpu_ops_base` — the base commit of the working branch cloned into
  `third_party/torch-xpu-ops` per AGENTS.md dev-override. Typically
  the `torch-xpu-ops` branch's `origin/main` (or its own `<base>`
  if pinned).

Pipeline for each sub-bug:

1. `fix/root-cause` (full — root_cause, fix_strategy, target_repo,
   domain, verdict). This step resolves `target_repo` for the
   verdict-table classification and for choosing `target_repo_dir`
   below.
2. Load domain skill via the registry (see
   `fix/domains/README.md`).
3. Derive **this sub-bug's** `target_repo_dir` from Step 1's
   `target_repo` (same rule as `issue-handler` Stage 4).
4. `fix/implement` with `allow_skip=false`,
   `patch_proposal_mode=true`, and the derived `target_repo_dir` —
   the STRICT patch-acceptance rules apply exactly the same as in
   the bug branch: no skip/xfail/seed/tolerance workarounds, no
   assertion deletion, no broad `try/except` around the failing
   call.
5. `fix/verify` with the same `target_repo_dir`.
6. Fresh-context review subagent (same one `issue-handler`
   Stage 5.5 uses).
7. **Capture this sub-bug's patch NOW, before any reset:**
   ```bash
   sub_patch=$(git -C <target_repo_dir> diff --cached)
   # Also capture xpu.txt override state if torch-xpu-ops, so the
   # patch-proposal comment can note the pin update the human needs.
   ```
   Store `sub_patch` in this sub-bug's `sub_bugs[]` entry immediately.
8. **Reset BOTH checkouts before the next sub-bug:**
   ```bash
   # `checkout --force`, not `reset --hard`: this skill does not create
   # its own branches, so HEAD may still be on a branch another run
   # owns (e.g. agent/issue-<M> holding that issue's audit commit).
   # `reset --hard` would move that branch pointer and destroy it;
   # `checkout --force` detaches HEAD and leaves every branch intact
   # while still discarding staged/unstaged changes.
   git -C <pytorch_dir> checkout --force <pytorch_base>
   git -C <pytorch_dir> clean -fdx
   # git clean -fdx does NOT descend into nested git repositories
   # (third_party/torch-xpu-ops is preserved by git's default
   # nested-repo protection).
   #
   # Restore third_party/xpu.txt to its origin/main state so the
   # next sub-bug's rebuild starts from a clean pin.
   git -C <pytorch_dir> checkout -- third_party/xpu.txt

   if [ -d "<pytorch_dir>/third_party/torch-xpu-ops/.git" ]; then
       git -C <pytorch_dir>/third_party/torch-xpu-ops checkout --force <xpu_ops_base>
       git -C <pytorch_dir>/third_party/torch-xpu-ops clean -fdx
   fi
   ```

Produce one **patch-proposal comment** per sub-bug. Each comment MUST
contain:

- The test's node id (uniquely identifies the sub-bug within the
  issue).
- Root cause (one paragraph, cites the specific line / symbol).
- Verified patch diff — or `NEEDS_HUMAN` verdict with concrete fix
  location if no root-cause fix is possible.
- Reproducer command (from `fix/reproduce`'s `refined_command`).
- Verify output — before (failing) and after (passing).
- `git apply` instructions.

Do NOT bundle multiple sub-bug fixes into a single mega-diff — one
comment per test keeps the audit trail clean and lets the maintainer
review, cherry-pick, or reject each independently.

`ALREADY_FIXED`, `ENVIRONMENT`, and `FLAKY` entries need NO
patch-proposal comment — the verdict table already tells the
maintainer to remove or ignore the skip.

### Step 6 — Assemble verdict table and outcome

Now that every `STILL_FAILING` entry has a `target_repo` from its
Step 5 triage, assemble the verdict table:

```
| Test | Classification | Notes |
|---|---|---|
| test/xpu/test_foo.py::TestBar::test_baz | STILL_FAILING_XPU_BUG | see sub-bug comment below |
| test/xpu/test_qux.py::TestBar::test_quux | ALREADY_FIXED | passes on nightly <version> |
| ... | ... | ... |
```

- `STILL_FAILING` + `target_repo == "pytorch"` → `STILL_FAILING_UPSTREAM_BUG`
- `STILL_FAILING` + `target_repo == "torch-xpu-ops"` → `STILL_FAILING_XPU_BUG`

This table is required even if every entry is `ALREADY_FIXED` — it is
the maintainer-facing record of "which skips can be removed".

Outcome selection:

- Every sub-bug is `PATCH_PROPOSED` (verified) OR every entry is
  `ALREADY_FIXED` / `ENVIRONMENT` / `FLAKY` → outcome
  `DONE_SKIP_TRIAGED`.
- One or more sub-bugs landed `NEEDS_HUMAN` → outcome
  `SKIP_TRIAGED_NEEDS_HUMAN`. The maintainer sees both the verdict
  table and per-sub-bug patch-proposal comments and knows exactly
  which sub-bugs remain.

**Never modify the issue body** regardless of outcome — same hard
rule as `issue-handler`. All output goes into GitHub comments.

## Output

Return to the orchestrator:

```json
{
  "verdict_table": [
    {
      "test": "<pytest node id>",
      "classification": "STILL_FAILING_UPSTREAM_BUG | STILL_FAILING_XPU_BUG | ALREADY_FIXED | ENVIRONMENT | FLAKY",
      "notes": "<one-line context, e.g. wheel version for ALREADY_FIXED>"
    }
  ],
  "sub_bugs": [
    {
      "test": "<pytest node id>",
      "status": "PATCH_PROPOSED | NEEDS_HUMAN",
      "root_cause": "<one paragraph>",
      "target_repo": "pytorch | torch-xpu-ops",
      "domain": "<domain from registry>",
      "reproducer_command": "<refined pytest command>",
      "patch_diff": "<unified diff, or empty if NEEDS_HUMAN>",
      "verify_before": "<pytest output before fix>",
      "verify_after": "<pytest output after fix, or empty if NEEDS_HUMAN>",
      "reason": "<one-line reason, especially for NEEDS_HUMAN>"
    }
  ],
  "outcome": "DONE_SKIP_TRIAGED | SKIP_TRIAGED_NEEDS_HUMAN"
}
```

The orchestrator is responsible for turning `verdict_table` and each
entry of `sub_bugs` into GitHub comments and setting the state
comment / labels.

## HARD RULES

- NEVER modify the issue body.
- NEVER bundle multiple sub-bug fixes into one diff.
- NEVER apply forbidden workarounds (skip/xfail/seed/tolerance) — the
  STRICT patch-acceptance rules from `fix/implement` Step 3.5 apply
  unchanged.
- NEVER conclude "already fixed" from a skip decorator existing — a
  skip confirms the issue existed at some point, not that it's fixed
  now.
- Struck-through entries (`~~...~~`) are already-resolved; dropping
  them silently is correct (they are not sub-bugs).
