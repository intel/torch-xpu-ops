---
name: issue-handler
description: >
  End-to-end orchestrator for fixing a single GitHub issue on pytorch or
  torch-xpu-ops. Sequences fix/ leaf skills into a pipeline and reports the
  result to the user or as GitHub comments on the issue (never modifies
  the issue body).
---

# Issue Handler — Orchestrator

Sequences `fix/reproduce`, `fix/triage`, `fix/implement`, `fix/verify`, and
a fresh-context **review subagent** into a single pipeline for one GitHub
issue. Leaf-skill logic lives in those files; this skill owns the
scheduling, mode handling, review-loop orchestration, and reporting.

## Execution modes

Decide mode once at the start and keep it for every stage.

- **Interactive (default):** human present. Ask when blocked. Report
  conversationally. **Never modify the GitHub issue body.** All output
  goes to the user in the chat.
- **Pipeline:** automated. No human to ask. **Never modify the GitHub
  issue body.** Post status, verdicts, diffs, logs, etc. as GitHub
  **comments only**. Apply `agent:*` labels for machine-readable state.

**HARD RULE: the issue body is user-owned and read-only for this skill.**
The skill must never `gh issue edit --body` (or any equivalent) on any
issue it processes. Reproducers, verdicts, patch proposals, status
markers, log slots — everything goes into GitHub comments. The only
issue-level state changes allowed are label additions/removals from the
`agent:*` namespace and (in a future extension) closing the issue when
the user has explicitly authorized it.

If a prior pipeline run modified an issue body, `git diff` the original
back from the issue-timeline API or from local snapshots in
`agent_space_xpu/session*/issue_N.json` and restore it before doing any
new work.

### Pipeline mode + multiple issues: use tmux

When pipeline mode is invoked with **more than one issue**, load the
`tmux-long-tasks` skill and run the pipeline for each issue inside its
own tmux window before doing anything else. Reasons:

- The bash tool kills processes on timeout. A single issue's pipeline
  (reproduce -> triage -> implement -> verify -> review) commonly runs
  10-30 minutes on nightly-wheel path, and much longer if a source
  build is involved. Running N issues inline serializes them under one
  bash timeout and risks losing everything on interrupt.
- tmux windows survive `/clear`, terminal disconnects, and the agent's
  own context limits. Each window's log persists to disk.
- Windows can be checked with a single `tmux capture-pane -p` — cheap
  to poll, cheap to resume.

Default layout:

- One tmux session (reuse if present): `xpu_fix` or a name the user
  chose.
- One window per issue: `issueN` (e.g. `issue1`, `issue2`).
- Each window's command tees to
  `agent_space_xpu/issueN_pipeline.log`.
- The main agent polls windows serially (one at a time) unless the user
  explicitly asks for parallel execution across issues.

Exceptions (do NOT force tmux):

- Interactive mode with a single issue.
- Pipeline mode with exactly one issue whose reproducer is expected to
  take under ~2 minutes and the user has not asked for tmux.
- User explicitly says "no tmux" / "run inline".

### Pipeline mode: comment contract

Since the issue body is read-only (see the hard rule above), all
pipeline state is expressed via **labels + a single "state comment"**.

**State machine** (advance the state comment through these stages;
labels track the terminal-ish stages):

```
DISCOVERED -> UPSTREAM_VERIFYING -> WAITING_UPSTREAM -> TRIAGING ->
TRIAGED -> IMPLEMENTING -> IN_REVIEW -> PUBLIC_PR -> CI_WATCH -> MERGED
```

Terminal stages: `DONE`, `SKIPPED`, `NEEDS_HUMAN`, `PATCH_PROPOSED`,
`DONE_SKIP_TRIAGED`.

Stage → label mapping:

| Stage(s) | Label |
|---|---|
| DISCOVERED, UPSTREAM_VERIFYING, TRIAGING, IMPLEMENTING, IN_REVIEW, PUBLIC_PR, CI_WATCH, MERGED | `agent:active` |
| WAITING_UPSTREAM | `agent:waiting-upstream` |
| TRIAGED, PATCH_PROPOSED | `agent:triaged` |
| DONE, SKIPPED, DONE_SKIP_TRIAGED | `agent:done` |
| NEEDS_HUMAN | `agent:needs-human` |

**Single state-comment pattern.** Keep exactly one machine-readable
"state comment" per issue. On the first pipeline run, post it as a new
comment starting with a fenced marker:

````
<!-- agent:state -->

## Agent pipeline status: <STAGE>

- **Handler:** issue-handler (pipeline)
- **Stage:** <STAGE>
- **Reproduced:** yes / no / cannot-verify (+ command)
- **Root cause:** <one sentence, if triaged>
- **Reviewer verdict:** APPROVE / REQUEST_CHANGES / BLOCK / not-run
- **Outcome:** <IMPLEMENTING | PATCH_PROPOSED | ...>

<details><summary>Discovery log</summary>...</details>
<details><summary>Env log</summary>...</details>
<details><summary>Upstream log</summary>...</details>
<details><summary>Triage log</summary>...</details>
<details><summary>Fix log</summary>...</details>
<details><summary>Verification log</summary>...</details>
<details><summary>Review log</summary>...</details>

*Automated by issue-handler.*
````

On subsequent runs, find the existing state comment by the
`<!-- agent:state -->` marker (via
`gh issue view N --comments --json comments -q '.comments[]'`) and
**edit it in place** (`gh issue comment --edit-last` is not enough
because the state comment may not be the last one; use the API:
`gh api /repos/<owner>/<repo>/issues/comments/<id> -X PATCH -f body=@file`).

**Additional comments** (patch proposals, per-test verdict tables,
etc.) are posted as separate comments and linked from the state
comment.

## Inputs

- A GitHub issue URL, number, or raw body on `pytorch` or `torch-xpu-ops`.
- Local checkout and Python environment for reproduction/fix stages.
- `pr_repo` (optional) — the ONE repo this run is allowed to open a PR
  against. Default: **the repo that hosts the issue**. Any other repo,
  even if triage decides the fix belongs there, is patch-proposal only:
  the diff is written to the issue and a human decides whether to open a
  follow-up PR after review. Accepted explicit values: `pytorch`,
  `torch-xpu-ops`, or `none` (never open a PR on this run — everything is
  patch-proposal).

## Pipeline

```
issue-format → reproduce → triage → implement → verify → review → report
```

### Stage 1 — issue-format

Classify as `bug`, `skip-list`, or `nonbug` and extract metadata.

- `nonbug` → record classification, report to user, **stop**.
- `skip-list` (Bug Skip template — a list of already-skipped tests asking
  "should these still be skipped?") → route to the **skip-triage branch**:
  1. Parse the test entries from the body. Struck-through entries
     (`~~...~~`) are already-resolved; skip them.
  2. For each remaining entry, call `fix/reproduce` with just that test as
     the reproducer.
  3. Classify each result:
     - `REPRODUCED` (FAILED) with root cause in pytorch → `STILL_FAILING_UPSTREAM_BUG`
     - `REPRODUCED` (FAILED) with root cause in torch-xpu-ops → `STILL_FAILING_XPU_BUG`
     - `NOT_REPRODUCED` (PASSED on nightly / source build) → `ALREADY_FIXED`
     - `NO_REPRODUCER` (test does not exist / `collected 0 items`) or
       `CANNOT_VERIFY` due to test-name drift → `ENVIRONMENT`
     - Intermittent (pass on retry) → `FLAKY`
  4. Do NOT call `fix/implement`. Post the per-test verdict table as an
     issue comment.
  5. Outcome is `DONE_SKIP_TRIAGED`; apply `agent:done` label and set
     status marker to `<!-- agent:status:DONE -->` with `(skip-triaged)`
     suffix in the body summary.
- `bug` → continue to Stage 2.

### Stage 2 — fix/reproduce

Call `fix/reproduce` with:
- `reproducer_command` from the issue body (if present)
- `ci_commit` if the issue references a specific CI run
- `pytorch_dir` if available; otherwise `fix/reproduce` clones to
  `agent_space_xpu/pytorch/`

Interpret the output:

| Output | Action |
|--------|--------|
| `REPRODUCED` | Continue to Stage 3 with `refined_command` |
| `NOT_REPRODUCED` | Triage to collect why; report to user; stop |
| `NO_REPRODUCER` | Continue to Stage 3 (static triage only) |
| `CANNOT_VERIFY` | Report blocker to user; stop |

### Stage 3 — fix/triage

Call `fix/triage` with the failure description (error log, context, and
`refined_command` if available from Stage 2).

Triage returns `target_repo` (`pytorch` or `torch-xpu-ops`) alongside the
verdict. Compare it to `pr_repo`:

| Verdict | `target_repo` vs `pr_repo` | Action |
|---------|----------------------------|--------|
| `IMPLEMENTING` | `target_repo == pr_repo` | Continue to Stage 3.5 (normal path — will end in a PR) |
| `IMPLEMENTING` | `target_repo != pr_repo` | Continue to Stage 3.5 in **patch-proposal mode** — implement + verify locally in `target_repo`'s checkout, but Stage 6 writes the diff to the issue instead of opening a PR |
| `NEEDS_HUMAN` | any | Report reason to user; stop |

Patch-proposal mode is the "cross-repo" path: this run is allowed to touch
files in `target_repo`'s local checkout to produce and verify a concrete
patch, but is NOT allowed to open a PR there. Deliverable is the diff
posted to the issue for human review. A follow-up PR (if warranted) is a
separate decision made by the reviewer, not by this skill.

### Stage 3.5 — Load domain skill

Read the `domain` field from the triage output. Use the skill tool to load
`fix/domains/<domain>` (e.g. `fix/domains/xpu-kernel`). If no domain skill
exists for the reported domain, proceed without it.

### Stage 4 — fix/implement

Call `fix/implement` with:
- `triage_result` from Stage 3
- `pytorch_dir`
- `allow_skip=false` — issue-handler never allows adding skip decorators
- no `commit_message_template` (use standard format)

In patch-proposal mode (Stage 3 chose it), additionally:
- Instruct `fix/implement` to leave changes **staged but uncommitted** in
  `target_repo`'s working tree. Stage 6 will read them back via
  `git -C <target_repo_dir> diff --cached`.
- Do NOT invoke any PR-creation skill later. The deliverable is the diff
  on the issue, not a branch.

### Stage 5 — fix/verify

Call `fix/verify` with:
- `refined_command` from Stage 2
- `pytorch_dir`
- `changed_files` from Stage 4
- `run_before_after_diff=false`
- `run_lint=false`

Note: if you ever set `run_before_after_diff=true` here, `fix/implement` must
leave changes staged but uncommitted (its default contract) — do not commit
before calling verify.

| Output | Action |
|--------|--------|
| `PASSED` | Continue to Stage 5.5 |
| `FAILED` | Loop back to Stage 4 with failure output (max 3 attempts) |
| `CANNOT_VERIFY` | Report to user; stop |

If 3 attempts exhausted without `PASSED`, report `NEEDS_HUMAN`.

### Stage 5.5 — Review subagent

Once `fix/verify` returns `PASSED`, spawn a **new subagent** with fresh
context to review the change. This is a gatekeeper step, mirroring the
`fix-issue` skill in pytorch: the implementer must not review its own
work. Skipping this stage is not allowed.

Use the `Task` tool with `subagent_type=general`. Pass the reviewer:

- The GitHub issue body and comments (raw).
- The verified `refined_command` and its output from Stage 5.
- The diff produced by Stage 4:
  `git -C <target_repo_dir> diff --cached`.
- The current `target_repo` and `pr_repo` (so it knows whether Stage 6
  will open a PR or post a patch proposal).

Instruct the reviewer to (this is the reviewer's checklist, not the
orchestrator's):

1. Read the issue body / comments for context on the bug being fixed.
2. Read `git diff --cached` and verify the changes fix the **root
   cause**. Flag any hack or workaround that dodges the real cause.
3. Confirm the diff is minimal and scoped: every changed line traces
   back to the triage output. Flag unrelated churn.
4. Confirm no debug prints, `TODO`/`FIXME` markers, commented-out code,
   or leftover experiment scaffolding.
5. Flag overly broad `try/except:` blocks that hide bugs.
6. Flag overly defensive `getattr` / `hasattr` checks that should be
   base-class schema updates instead.
7. Confirm no untracked files. All intended changes are staged; nothing
   extraneous is staged.
8. If the diff touches tests, confirm test tolerances/skips are
   consistent with the failure mode (see `fix/reproduce`'s
   "Use the test's own assertion" rule).
9. Apply relevant rules from `.claude/skills/pr-review/` if present.

The reviewer returns one of:

- `APPROVE` — diff is ready. Continue to Stage 6.
- `REQUEST_CHANGES` — specific issues to address. Loop back to Stage 4
  with the reviewer's feedback appended to `triage_result`. Do NOT
  re-run Stage 3 unless the reviewer says triage itself is wrong.
- `BLOCK` — fundamental problem that Stage 4 can't fix (e.g. the bug is
  actually intended behavior, or the fix requires cross-repo redesign).
  Stop with `NEEDS_HUMAN` and include the reviewer's reason.

**Review loop cap:** 2 review passes. If the second review still
returns `REQUEST_CHANGES`, stop with `NEEDS_HUMAN` and include both
rounds of feedback in the report — do not enter a third round.

**Patch-proposal mode:** run the reviewer just as strictly. A
patch-proposal is a diff a human will apply upstream; sloppy diffs
waste reviewer time on the upstream side.

### Stage 6 — Report and hand off

Summarize the outcome. In **interactive mode**, report to the user. In
**pipeline mode**, update the machine-readable state comment (single
comment per issue, marked with `<!-- agent:state -->`) and post any
extra deliverables (patch proposals, verdict tables) as additional
comments. **Do not modify the issue body.**

Always include:
- Issue link and one-line title
- Classification (bug/nonbug + category)
- Reproduced: yes / no / cannot-verify (+ command used)
- Root cause (one sentence)
- Files changed (or "none" + reason)
- Fix verified: PASS / FAIL / not-attempted (+ command)
- Reviewer verdict: APPROVE / REQUEST_CHANGES / BLOCK / not-attempted
  (+ round count if looped)
- Outcome: `IMPLEMENTING` / `PATCH_PROPOSED` / `DONE_SKIP_TRIAGED` /
  `NEEDS_HUMAN` / `SKIPPED` / `NOT_REPRODUCED`

Routing by `target_repo` vs `pr_repo` (see Stage 3):

- `target_repo == pr_repo` and fix verified -> outcome `IMPLEMENTING`. Hand
  off to the PR-creation skill for `pr_repo`
  (`xpu-ops-pr-creation` for `torch-xpu-ops`). Do not open the PR from this
  skill.
- `target_repo != pr_repo` and fix verified -> outcome `PATCH_PROPOSED`.
  Do NOT open a PR anywhere. Instead, post a "patch proposal" comment on
  the issue with:
  - Target repo (`target_repo`) and a one-line rationale for why the fix
    lives there
  - For each changed file: absolute repo-relative path + a fenced
    ```diff block of `git diff` output (unified format)
  - Reproducer command and verification result
  - A "how to apply" line, e.g. `cd <target_repo>; git apply <<'EOF' ...`
  - Advance the state comment to stage `TRIAGED` (or `PATCH_PROPOSED`)
    and apply the `agent:triaged` label. The pipeline stops here for
    this issue; a human takes the diff to `target_repo`.
- `NEEDS_HUMAN` -> outcome `NEEDS_HUMAN`, apply `agent:needs-human`.

The diff MUST come from an actual verified **and reviewed** change on
disk (Stage 4 -> Stage 5 PASSED -> Stage 5.5 APPROVE). Do not post
speculative diffs. If verify did not pass, outcome is `NEEDS_HUMAN`,
not `PATCH_PROPOSED`. If review did not approve after 2 passes, outcome
is `NEEDS_HUMAN`, not `PATCH_PROPOSED`.

## Iterative loop

The pipeline is not strictly linear. Loop when a later stage invalidates an
earlier assumption:

- Stage 5 FAILED → return to Stage 4 (refine the fix)
- Stage 5.5 REQUEST_CHANGES → return to Stage 4 (address reviewer feedback)
- Stage 5.5 BLOCK → stop with `NEEDS_HUMAN`
- Stage 4 reveals triage was wrong → return to Stage 3
- Stage 3 finds reproducer is wrong → return to Stage 2

Soft caps:
- 3 fix attempts triggered by verify failure (Stages 4-5).
- 2 review passes triggered by reviewer `REQUEST_CHANGES` (Stages 4-5.5).

Stop with `NEEDS_HUMAN` when either cap is hit.

## HARD RULES

- **Never modify the GitHub issue body.** In either interactive or
  pipeline mode. All output goes to comments (or, in interactive mode,
  the chat). If a prior run of this skill (or any tool called by it)
  wrote to the body, restore the body from the timeline API or from
  local snapshots before doing anything else.
- **`agent:triaged` label requires a real `fix/triage` run.** Do not
  apply that label from a stage that only ran `fix/reproduce`.
- **`PATCH_PROPOSED` requires** Stage 4 -> Stage 5 PASSED -> Stage 5.5
  APPROVE. No speculative diffs.
- **Do not open a PR on any repo other than `pr_repo`.** If
  `target_repo != pr_repo`, use patch-proposal only. The default
  `pr_repo` is the repo that hosts the issue.
- **Do not commit in `patch_proposal_mode`.** `fix/implement` leaves
  changes staged.
