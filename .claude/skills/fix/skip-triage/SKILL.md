---
name: fix/skip-triage
description: >
  Process a "Bug Skip" tracking issue — a list of already-skipped tests
  asking whether each skip is still needed. For each entry: re-verify
  via fix/reproduce, classify (still-failing / already-fixed / env /
  flaky), and for still-failing entries run the full fix pipeline as a
  sub-bug producing one patch-proposal per test. Called by issue-handler
  Stage 1 when issue-format returns issue_type=skip-list.
---

# Skip-Triage — Bug Skip Issue Handler

Mini-orchestrator for skip-list issues. Loops `fix/reproduce` over each
test entry, then recursively runs the full bug pipeline
(`fix/triage` → `fix/implement` → `fix/verify` → review) for
still-failing entries. Peer of `fix/triage` (not a child of it); the
top-level orchestrator dispatches to this skill instead of running
the bug pipeline directly when the issue is a skip-list.

## Inputs

- `issue_body` — full GitHub issue body containing the test-entry
  checklist. Entries marked `~~strike-through~~` are already-resolved
  and MUST be skipped.
- `pytorch_dir` — local pytorch checkout path.
- `pr_repo` — the ONE repo this run is allowed to open a PR against.
  In skip-triage the deliverable is always patch-proposals on the
  issue, not PRs — so this skill treats every sub-bug as
  patch-proposal mode regardless of `pr_repo`. The orchestrator still
  passes `pr_repo` so future work can lift this restriction if
  needed.

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
the per-entry `fix/triage` run in Step 5. The verdict table (assembled
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
**sub-bug** and run the full bug pipeline in patch-proposal mode:

1. `fix/triage` (full — root_cause, fix_strategy, target_repo,
   domain, verdict). This step resolves `target_repo` for the
   verdict-table classification.
2. Load domain skill via the registry (see
   `fix/domains/README.md`).
3. `fix/implement` with `allow_skip=false` and
   `patch_proposal_mode=true` — the STRICT patch-acceptance rules
   apply exactly the same as in the bug branch: no skip/xfail/seed/
   tolerance workarounds, no assertion deletion, no broad `try/except`
   around the failing call.
4. `fix/verify`.
5. Fresh-context review subagent (same one `issue-handler` Stage 5.5
   uses).

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
