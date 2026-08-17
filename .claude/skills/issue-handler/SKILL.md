---
name: issue-handler
description: >
  End-to-end orchestrator for fixing a single GitHub issue on pytorch or
  torch-xpu-ops. Sequences fix/ leaf skills into a pipeline and reports the
  result to the user or as GitHub comments on the issue (never modifies
  the issue body).
---

# Issue Handler — Orchestrator

Sequences `issue-triage`, `fix/reproduce`, `fix/root-cause`, `fix/implement`,
`fix/verify`, and a fresh-context **review subagent** into a single pipeline
for one GitHub issue, then reports the outcome. For skip-list issues, Stage 1
dispatches to `fix/skip-list` instead of the bug pipeline. Leaf-skill logic
lives in those files; this skill owns the scheduling, mode handling,
review-loop orchestration, and reporting. See the "Pipeline" section below
for the full stage list.

## Prerequisites

### Shell helpers used throughout this skill

Every bash recipe in this file assumes two helper functions are in
scope. They are not shell builtins; define them once at the top of
the orchestrator's shell (or source a shared helpers file):

```bash
abort()    { echo "ABORT: $*" >&2; exit 1; }
log_warn() { echo "WARN: $*"  >&2; }
```

`abort` exits non-zero with a diagnostic; `log_warn` writes a warning
and returns success. Recipes below use them without redefining. The
same two helpers are assumed by `fix/verify`, `fix/skip-management`,
and `xpu-nightly-ci-fix`.

### GitHub CLI

This skill shells out to the `gh` CLI at every stage (fetch issue, post
comments, apply labels, edit the state comment). Authenticate `gh` **before**
invoking the skill — it does not prompt for credentials.

Required GitHub token scopes:

| Mode        | Classic PAT scope                  | Fine-grained equivalent                                        |
|-------------|-------------------------------------|-----------------------------------------------------------------|
| Interactive | `repo` (min `read:issues`)          | Issues: **Read**, Metadata: Read                                |
| Pipeline    | `repo` (must include `write:issues`)| Issues: **Read & Write**, Metadata: Read, Pull requests: Read   |

Pipeline mode additionally needs write access to add/remove `agent:*` labels,
post comments (Stage 5.5, Stage 6), and `PATCH` the state comment via
`gh api /repos/<owner>/<repo>/issues/comments/<id>`. A read-only token
silently degrades pipeline mode; if `write:issues` is missing, fall back to
interactive.

Preflight at Stage 0:

```bash
# The orchestrator MUST decide execution mode (see "Execution modes"
# below) before running this preflight. Set MODE accordingly:
#   MODE=interactive   # human present
#   MODE=pipeline      # automated
MODE=${MODE:-interactive}

# 1. Confirm gh is authenticated.
gh auth status 2>&1 | grep -q "Logged in to github.com" \
  || abort "gh not authenticated; run: gh auth login"

# 2. Extract the scope list and check for a write-capable scope.
#    Interactive mode requires 'repo' (or at minimum 'read:issues').
#    Pipeline mode requires 'repo' or 'write:issues' (write access
#    to issues + PRs).
scopes=$(gh auth status -t 2>&1 | grep -oE "'[a-z:_-]+'" | tr -d "'")
case "$MODE" in
  pipeline)
    echo "$scopes" | grep -qE '^(repo|write:issues)$' \
      || abort "pipeline mode requires 'repo' or 'write:issues' scope; got: $scopes"
    ;;
  interactive)
    echo "$scopes" | grep -qE '^(repo|read:issues|write:issues)$' \
      || abort "interactive mode requires at least 'read:issues' scope; got: $scopes"
    ;;
  *)
    abort "unknown MODE='$MODE' (expected 'interactive' or 'pipeline')"
    ;;
esac

# 3. Every subsequent gh call in this skill MUST check its exit code.
#    A non-zero exit from any 'gh issue comment', 'gh issue edit',
#    'gh api ... PATCH' is treated as pipeline failure (retry once,
#    then NEEDS_HUMAN).
```

If either check fails, do not enter the pipeline. Emit `NEEDS_HUMAN` naming
the missing scope. User fixes with:

```bash
gh auth refresh -s repo
# or regenerate a fine-grained PAT with Issues: Read & Write
# https://github.com/settings/personal-access-tokens
```

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

### Pipeline mode: per-issue branch isolation

When pipeline mode processes multiple issues, each issue's fix lives on
its own git branch `agent/issue-<N>` (`<N>` is the GitHub issue number)
in the `target_repo` checkout, so runs do not stomp each other.

Sequence:

- **Before Stage 4**, branch off the base **of `target_repo`'s own
  checkout**. The two repos have distinct commit graphs — a pytorch sha
  does not exist in torch-xpu-ops — so `<base>` is chosen by
  `target_repo`, not by which base `fix/reproduce` reported:

  | `target_repo` | `<base>` |
  |---|---|
  | `pytorch` | `origin/main`, or `<ci_commit_sha>` when `fix/reproduce` reported `base=<ci_commit_sha>` (fallback path used because trunk failed to build). The whole pipeline stays on it through Stage 5 so verify's rebuild also succeeds. |
  | `torch-xpu-ops` | `<xpu_ops_base>` — the base of the working branch cloned into `third_party/torch-xpu-ops` per AGENTS.md "Commit Pin & Development Override" (that repo's own `origin/main` unless the caller pinned it). Never a pytorch sha. |

  ```bash
  git -C <target_repo_dir> checkout -b agent/issue-<N> <base>
  ```

  For `target_repo=pytorch`, `fix/reproduce` Stage 2 leaves HEAD on that
  base at exit, so this is just a `checkout -b` from where reproduce
  stopped — no history is lost. `-b` (not `-B`) so a pre-existing
  `agent/issue-<N>` from a prior run fails loudly instead of silently
  discarding its history.
  If it exists, decide explicitly: reuse it
  (`git checkout agent/issue-<N>`) or delete it
  (`git branch -D agent/issue-<N>`) after logging what gets thrown
  away.

- **Stages 4, 5, 5.5 run on this branch.** Changes stay staged and
  uncommitted throughout (that is `fix/implement`'s contract and what
  the reviewer and patch-proposal reads via `git diff --cached`). Do
  not commit here.

- **After Stage 6 decides the outcome, commit once** on
  `agent/issue-<N>`:

  | Outcome | Commit message |
  |---|---|
  | `IMPLEMENTING` (PR path)      | Real fix message (see AGENTS.md) |
  | `PATCH_PROPOSED`              | `WIP: patch proposed for #<N>` |
  | `NEEDS_HUMAN` after Stage 5/5.5 | `WIP: needs human — <reason>` |

  Every terminal state commits. The branch is the audit trail; a
  branch with no commit is not an audit trail. For `PATCH_PROPOSED`
  the commit happens only **after** Stage 6 has read the staged diff
  and posted it to the issue, and the branch is never pushed.

- **Then switch away** before starting the next issue:

  ```bash
  git -C <target_repo_dir> checkout main   # or wherever reproduce left it
  ```

  The next issue's Stage 4 will `checkout -b agent/issue-<M>` from
  there.

- **PR handoff.** For `IMPLEMENTING`, `xpu-ops-pr-creation` is invoked
  with `agent/issue-<N>` already checked out and containing the fix
  commit. That skill handles push, PR body, and any branch renaming it
  needs. issue-handler does not push and does not rename.

- **Never run `fix/implement` from `main` or another issue's branch.**
  If the current branch is wrong at Stage 4 entry, abort with
  `NEEDS_HUMAN`.

### Pipeline mode: comment contract

Since the issue body is read-only (see the hard rule above), all
pipeline state is expressed via **labels + a single "state comment"**.

**State machine** (advance the state comment through these stages;
labels track the terminal-ish stages):

```
DISCOVERED -> UPSTREAM_VERIFYING -> WAITING_UPSTREAM -> TRIAGING ->
TRIAGED -> IMPLEMENTING -> IN_REVIEW -> PUBLIC_PR -> CI_WATCH -> MERGED
```

Terminal stages: `DONE`, `NOT_REPRODUCED`, `NEEDS_HUMAN`,
`PATCH_PROPOSED`, `DONE_SKIP_TRIAGED`, `SKIP_TRIAGED_NEEDS_HUMAN`.

There is no `SKIPPED` terminal stage: this orchestrator always runs
`fix/implement` with `allow_skip=false`, so it can never end a run by
adding a skip decorator. A stage that cannot decide (`CANNOT_VERIFY`
from `fix/reproduce` or `fix/verify`) terminates as `NEEDS_HUMAN` with
the blocker as the reason — see Stage 6.

Stage → label mapping:

| Stage(s) | Label |
|---|---|
| DISCOVERED, UPSTREAM_VERIFYING, TRIAGING, IMPLEMENTING, IN_REVIEW, PUBLIC_PR, CI_WATCH, MERGED | `agent:active` |
| WAITING_UPSTREAM | `agent:waiting-upstream` |
| TRIAGED, PATCH_PROPOSED | `agent:triaged` |
| DONE, NOT_REPRODUCED, DONE_SKIP_TRIAGED | `agent:done` |
| NEEDS_HUMAN, SKIP_TRIAGED_NEEDS_HUMAN | `agent:needs-human` |

`agent:active` is applied by Stage 1 on entry (pipeline mode only),
with the retry-then-warn rule below; every terminal stage replaces it
via `apply_terminal_label`.

**Label operation error handling.** Every `gh issue edit --add-label`
or `--remove-label` call MUST check its exit code. Failure modes:

- Rate limit / transient network → retry once with a 5s sleep.
- Second retry still fails **and** the label being applied is a
  terminal-state label (`agent:done`, `agent:needs-human`,
  `agent:triaged`) → downgrade the pipeline outcome to `NEEDS_HUMAN`,
  reason: `"failed to apply terminal label <name>: <gh stderr>"`.
  Post a comment on the issue with the same reason so the maintainer
  sees why the outcome does not match what the state comment claims.
- Second retry still fails **and** the label is a mid-flight state
  (`agent:active`, `agent:waiting-upstream`) → log a warning in the
  state comment `Discovery log` section but continue the pipeline;
  the terminal label at the end will still surface the outcome.

Silent-no-op is forbidden — the label state is what batch tooling and
issue filters rely on to find agent-owned issues.

**Label transitions on `agent:*`** — when applying a terminal label,
remove any prior non-terminal `agent:*` label first (e.g. moving from
`agent:active` → `agent:needs-human` must also remove `agent:active`).
This applies especially when `fix/root-cause` downgrades an
`agent-fixable` verdict to `NEEDS_HUMAN`: remove `agent:triaged` /
`agent:active` before adding `agent:needs-human`.

**Concrete `apply_terminal_label` subroutine** — every stage that
lands on a terminal outcome (Stage 2 NOT_REPRODUCED / CANNOT_VERIFY,
Stage 3 NEEDS_HUMAN, Stage 5 CANNOT_VERIFY or attempts exhausted,
Stage 5.5 BLOCK, Stage 6 for DONE / PATCH_PROPOSED / NEEDS_HUMAN /
DONE_SKIP_TRIAGED / SKIP_TRIAGED_NEEDS_HUMAN) MUST call this exact
sequence rather than issuing raw `--add-label` calls:

```bash
apply_terminal_label() {
    # Args: $1=owner/repo, $2=issue_number, $3=new_terminal_label
    local repo="$1" n="$2" new_label="$3"
    # Terminal labels this pipeline may apply. Anything else is a bug.
    case "$new_label" in
      agent:done|agent:needs-human|agent:triaged) ;;
      *) abort "apply_terminal_label: '$new_label' is not a terminal label" ;;
    esac

    # Short-circuit: if the label is already applied, we're done. Avoids
    # a churn of remove + failed re-add on pipeline retries.
    if gh issue view "$n" --repo "$repo" --json labels \
         --jq '.labels[].name' 2>/dev/null | grep -qx "$new_label"; then
        return 0
    fi

    # Remove only NON-terminal `agent:*` labels. Do NOT strip other
    # terminal labels — a concurrent orchestrator run may have just
    # applied one legitimately, and the caller is responsible for
    # explicit terminal-to-terminal transitions.
    for stale in agent:active agent:waiting-upstream; do
        _gh_edit_label_with_retry "$repo" "$n" --remove-label "$stale" \
            "remove-label $stale"
    done
    # Add the new terminal label with the same retry policy.
    _gh_edit_label_with_retry "$repo" "$n" --add-label "$new_label" \
        "add-label $new_label" \
        || abort "apply_terminal_label: failed to add '$new_label'"
}

# Helper: retry a single `gh issue edit` label mutation, distinguishing
# transient failures (which retry once) from non-retryable ones
# ("already has label", "label not found", "not found in list") which
# are treated as success.
_gh_edit_label_with_retry() {
    local repo="$1" n="$2" op_flag="$3" label="$4" description="$5"
    local out rc
    for attempt in 1 2; do
        out=$(gh issue edit "$n" --repo "$repo" "$op_flag" "$label" 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then
            return 0
        fi
        # Non-retryable, no-op-equivalent errors: treat as success.
        case "$out" in
            *"already has label"*|*"already exists"*|*"not found in list"*|*"label not found"*)
                return 0
                ;;
        esac
        [ "$attempt" = 2 ] && { log_warn "$description failed: $out"; return $rc; }
        sleep 5
    done
}
```

Where the following stages call it:

- Stage 2 → `apply_terminal_label ... agent:done` on `NOT_REPRODUCED`,
  `... agent:needs-human` on `CANNOT_VERIFY`, before stopping.
- Stage 3 → `apply_terminal_label ... agent:needs-human` when
  `fix/root-cause` returns `NEEDS_HUMAN` (before "Report reason to
  user; stop").
- Stage 5 → `apply_terminal_label ... agent:needs-human` on
  `CANNOT_VERIFY` and when the 3 fix attempts are exhausted.
- Stage 5.5 → `apply_terminal_label ... agent:needs-human` on `BLOCK`.
- Stage 6 → dispatch by outcome:
  - `IMPLEMENTING` (ready for PR): keep `agent:active` (mid-flight);
    do NOT call `apply_terminal_label` — it will be applied when the
    PR opens or lands.
  - `PATCH_PROPOSED` → `apply_terminal_label ... agent:triaged`.
  - `DONE`, `NOT_REPRODUCED`, `DONE_SKIP_TRIAGED` →
    `apply_terminal_label ... agent:done`.
  - `NEEDS_HUMAN`, `SKIP_TRIAGED_NEEDS_HUMAN` →
    `apply_terminal_label ... agent:needs-human`.

Every stop path is a terminal outcome and therefore labels the issue.
An early stop (Stage 2 / Stage 3 / Stage 5) that leaves `agent:active`
applied is a pipeline bug — batch tooling would keep treating the issue
as in-flight forever.

Mid-flight label sets (`agent:active`, `agent:waiting-upstream`) are
applied by their originating stage with the retry-then-warn rule from
above; they are NOT terminal and do not go through this subroutine.

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
`<!-- agent:state -->` marker and edit it in place. `gh issue comment
--edit-last` is not enough because the state comment may not be the
last one — use the concrete recipe below:

```bash
OWNER=<owner> REPO=<repo> N=<issue_number>

# 1. Find the state-comment id by marker. Take the OLDEST match if
#    multiple exist (a duplicate can only appear if a prior run's
#    first post failed after write — dedupe by deleting the newer
#    ones later).
comment_id=$(gh issue view "$N" --repo "$OWNER/$REPO" \
  --json comments \
  --jq '.comments
        | map(select(.body | startswith("<!-- agent:state -->")))
        | sort_by(.createdAt)
        | .[0].id')

# 2. If no existing state comment, post a new one; else PATCH it.
if [ -z "$comment_id" ] || [ "$comment_id" = "null" ]; then
    gh issue comment "$N" --repo "$OWNER/$REPO" --body-file state.md
else
    # `--field` reads the value literally (no @file expansion); use
    # command substitution to inline the file body. `-f` is NOT
    # equivalent here — gh does not expand @path for POST/PATCH
    # fields the way curl does.
    gh api "/repos/$OWNER/$REPO/issues/comments/$comment_id" \
      --method PATCH \
      --field body="$(cat state.md)"
fi

# 3. If step 1 returned more than one id, delete the newer duplicates
#    so the next run finds a single unambiguous state comment.
#    Best-effort — a DELETE failure (403 on another user's comment,
#    422 on already-deleted, etc.) is logged but does NOT downgrade
#    the pipeline outcome; the state comment itself is intact.
gh issue view "$N" --repo "$OWNER/$REPO" --json comments \
  --jq '.comments
        | map(select(.body | startswith("<!-- agent:state -->")))
        | sort_by(.createdAt)
        | .[1:][] | .id' \
  | while read dup_id; do
      gh api "/repos/$OWNER/$REPO/issues/comments/$dup_id" \
        --method DELETE \
        || log_warn "could not delete duplicate state comment $dup_id; continuing"
    done
```

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
issue-triage → reproduce → root-cause → implement → verify → review → report
                 (Stage 1 dispatches to fix/skip-list instead when
                  issue_type == "skip-list")
```

### Stage 1 — issue-triage

Run `issue-triage` to classify the issue and extract shallow metadata
(`issue_type`, `runtime_dependencies`, `scope`, preliminary `verdict`).

In pipeline mode, apply `agent:active` before running it (state
`DISCOVERED`), using the retry-then-warn rule for mid-flight labels.
Every exit path below replaces it via `apply_terminal_label`.

Route on `verdict` and `issue_type` together. `issue_type=skip-list`
takes precedence over any `verdict` value — per-entry fixability is
what matters for skip-lists, decided by `fix/skip-list`:

- `issue_type == "skip-list"` (any `verdict`) → invoke `fix/skip-list`
  with `issue_body`, `pytorch_dir`, `pr_repo`, and the two base
  commits it resets to between sub-bugs (`pytorch_base`,
  `xpu_ops_base` — see that skill's Inputs). That skill owns the
  full skip-list pipeline (per-entry reproduce, classification,
  per-sub-bug fix pipeline in patch-proposal mode, verdict table +
  per-sub-bug patch-proposal outputs). When it returns:
  1. Post its `verdict_table` as a GitHub comment on the issue
     (mandatory deliverable — required even if every entry is
     `ALREADY_FIXED`).
  2. Post one patch-proposal comment per entry in its `sub_bugs`,
     using the fields `test`, `root_cause`, `patch_diff`,
     `reproducer_command`, `verify_before`, `verify_after`, and a
     `git apply` instruction line built from `patch_diff`. Skip
     `sub_bugs` entries whose status is `NEEDS_HUMAN` without a
     `patch_diff` — post them as NEEDS_HUMAN comments naming the
     reason and a concrete fix location instead.
  3. Set the state comment `Outcome` from `outcome`
     (`DONE_SKIP_TRIAGED` → `apply_terminal_label ... agent:done`;
     `SKIP_TRIAGED_NEEDS_HUMAN` → `apply_terminal_label ...
     agent:needs-human`).
  4. **Never modify the issue body** regardless of outcome.
- `verdict == "NEEDS_HUMAN"` and `issue_type != "skip-list"` → record
  classification and `reason`, report to user, `apply_terminal_label
  ... agent:needs-human`, **stop**. This covers non-bugs, umbrella
  tasks, and bugs with insufficient signal (no traceback, no
  reproducer, hardware-only, non-public deps). No Stage 2+ work.
- `verdict == "agent-fixable"`:
  - `issue_type == "nonbug"` — should not happen (nonbug forces
    `NEEDS_HUMAN` in `issue-triage`); if it does, treat as
    `NEEDS_HUMAN` defensively and stop.
  - `issue_type == "bug"` → carry `runtime_dependencies` and the
    preliminary `scope` forward into Stage 2/3 for reference, and
    continue to Stage 2. Do NOT stop on `scope == "both"` or
    `scope == "unclear"` here — those are preliminary and refined by
    `fix/root-cause` in Stage 3.

### Stage 2 — fix/reproduce

Call `fix/reproduce` with:
- `reproducer_command` from the issue body (if present)
- `ci_commit` if the issue references a specific CI run
- `pytorch_dir` if available; otherwise `fix/reproduce` clones to
  `agent_space_xpu/pytorch/`

Interpret the output:

| Output | Action |
|--------|--------|
| `REPRODUCED` | Continue to Stage 3 with `refined_command`. Record `stage` — a `stage=nightly` reproduction means the environment is a **wheel install**, and `fix/verify` requires a source build (see Stage 5) |
| `NOT_REPRODUCED` | Report to user with the reason `fix/reproduce` gave (which stage passed and what was checked); outcome `NOT_REPRODUCED`, `apply_terminal_label ... agent:done`; stop |
| `NO_REPRODUCER` | Continue to Stage 3 (static triage only). Stages 4-5 need a runnable command: if triage cannot name one, stop after Stage 3 with `NEEDS_HUMAN` (root cause + fix location reported, nothing implemented) |
| `CANNOT_VERIFY` | Report blocker to user; outcome `NEEDS_HUMAN` (reason: the reported `blocker`), `apply_terminal_label ... agent:needs-human`; stop |

### Stage 3 — fix/root-cause

Call `fix/root-cause` with the failure description (error log, context,
`refined_command` if available from Stage 2), and the preliminary
`scope` and `runtime_dependencies` from `issue-triage` as hints.

`fix/root-cause` reads source and produces the **final** `target_repo`,
`domain`, and `verdict`. It may override `issue-triage`'s preliminary
outputs:

- Preliminary `scope=unclear` → analysis resolves to a specific
  `target_repo`.
- Preliminary `scope=pytorch` or `scope=torch-xpu-ops` → analysis may
  confirm or (rarely) correct.
- Preliminary `scope=both` → analysis decides whether the fix can be
  isolated to one repo (single-repo `target_repo` output) or truly
  requires cross-repo changes (returns `NEEDS_HUMAN`, reason:
  cross-repo fix out of scope for this run).
- Preliminary `verdict=agent-fixable` → analysis may downgrade to
  `NEEDS_HUMAN` if source inspection reveals hidden complexity.
- Preliminary `verdict=NEEDS_HUMAN` never reaches this stage — Stage 1
  stops on that.

Compare `fix/root-cause`'s `target_repo` to `pr_repo`:

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

### Stage 3.5 — Load domain skill (via registry)

Consult `.claude/skills/fix/domains/README.md` — the domain registry —
before loading anything:

1. Read the `domain` field from the triage output.
2. Look it up in the registry's JSON list. If not present → **abort
   with `NEEDS_HUMAN`**, reason: `"fix/root-cause emitted domain not in
   fix/domains/README.md: <domain>"`.
3. Check the registry row: `skill_path` directory must exist. If not →
   **abort with `NEEDS_HUMAN`**, reason: `"registry lists <domain> but
   <skill_path> is missing"`.
4. Compare the row's `target_repo` with triage's `target_repo` output.
   Mismatch → **abort with `NEEDS_HUMAN`**, reason: `"triage
   target_repo=<x> conflicts with registry <y> for domain <domain>"`.
5. Only then, use the skill tool to load the `skill_path`.

Do NOT fall back to "proceed without a domain skill" — that silent
no-op is the bug the registry exists to prevent.

### Stage 4 — fix/implement

**Derive `target_repo_dir` from Stage 3's `target_repo`:**

- `target_repo == "pytorch"` → `target_repo_dir = pytorch_dir`.
- `target_repo == "torch-xpu-ops"` →
  `target_repo_dir = <pytorch_dir>/third_party/torch-xpu-ops`. The
  caller (or `xpu-build-pytorch` earlier in the pipeline) must have
  already cloned the working branch there per AGENTS.md "Commit Pin
  & Development Override"; if the directory is missing, abort with
  `NEEDS_HUMAN` reason `"third_party/torch-xpu-ops override checkout
  missing; run xpu-build-pytorch first"`.

In pipeline mode with multiple issues, switch to the per-issue branch
in `target_repo_dir` (see "Pipeline mode: per-issue branch isolation"
above for the full rules — including the fail-loud `-b` requirement
and `<base>` selection).

**First attempt:**

```bash
git -C <target_repo_dir> fetch origin
git -C <target_repo_dir> checkout -b agent/issue-<N> <base>
# <base> belongs to target_repo_dir's own history (see the base table in
# "Pipeline mode: per-issue branch isolation"): origin/main or
# <ci_commit_sha> for pytorch, <xpu_ops_base> for torch-xpu-ops.
# Use `-b` (not `-B`) so a stale branch fails loudly.
```

**Loop-back re-entry (from Stage 5 FAILED or Stage 5.5 REQUEST_CHANGES):**

```bash
# The agent/issue-<N> branch already exists and HEAD is on it — do NOT
# re-run `checkout -b` (that would fail loudly per the branch-isolation
# rule). Just confirm HEAD:
[ "$(git -C <target_repo_dir> rev-parse --abbrev-ref HEAD)" = "agent/issue-<N>" ] \
  || git -C <target_repo_dir> checkout agent/issue-<N>
# fix/implement's previous staged changes are still present; the loop
# refines them on top.
```

Then call `fix/implement` with:
- `triage_result` from Stage 3
- `pytorch_dir`
- `target_repo_dir` (derived above)
- `allow_skip=false` — issue-handler never allows adding skip decorators
- `patch_proposal_mode=<true|false>` — set to `true` if Stage 3 chose
  patch-proposal mode (`target_repo != pr_repo`), otherwise `false`

Stage 6 owns the commit message; `fix/implement` neither commits nor
takes a commit-message parameter.

In patch-proposal mode (Stage 3 chose it), additionally:
- `fix/implement` will leave changes **staged but uncommitted** in
  `target_repo_dir`'s working tree per its `patch_proposal_mode`
  contract. Stage 6 reads them back via
  `git -C <target_repo_dir> diff --cached`.
- Do NOT invoke any PR-creation skill later. The deliverable is the diff
  on the issue, not a branch.

### Stage 5 — fix/verify

If `fix/implement` returned `ready_for_verify: false` (Step 3.5 rejected
the diff and the implementer bailed with `NEEDS_HUMAN`), do NOT call
`fix/verify`. Skip directly to reporting `NEEDS_HUMAN` in Stage 6 with
the reviewer's citations.

**Source-build precondition.** `fix/verify` Step 1 refuses to run
against a wheel install, and a locally staged fix has no effect on an
installed wheel anyway. If Stage 2 reproduced at `stage=nightly` (no
source build was ever made), build before calling verify: load
`xpu-build-pytorch` and build `pytorch_dir` at `<base>` with the fix
staged. If that build fails, outcome is `NEEDS_HUMAN` (reason: the
build error), `apply_terminal_label ... agent:needs-human`; stop.

Otherwise call `fix/verify` with:
- `refined_command` from Stage 2
- `pytorch_dir`
- `target_repo_dir` (the same path derived in Stage 4 — verify's
  `git stash` / `git diff --cached` must run where the fix is staged)
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
| `CANNOT_VERIFY` | Report the `blocker` to user; outcome `NEEDS_HUMAN`, `apply_terminal_label ... agent:needs-human`; stop |

If 3 attempts exhausted without `PASSED`, report `NEEDS_HUMAN` and
`apply_terminal_label ... agent:needs-human`.

### Stage 5.5 — Review subagent

Once `fix/verify` returns `PASSED`, spawn a **new subagent** with fresh
context to review the change. This is a gatekeeper step, mirroring the
`fix-issue` skill in pytorch: the implementer must not review its own
work. Skipping this stage is not allowed.

Use the `Task` tool with `subagent_type=general-purpose`. Pass the reviewer:

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
  `SKIP_TRIAGED_NEEDS_HUMAN` / `NEEDS_HUMAN` / `NOT_REPRODUCED`

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
    ```diff block of `git -C <target_repo_dir> diff --cached` output
    (unified format; the fix is staged, so plain `git diff` is empty)
  - Reproducer command and verification result
  - A "how to apply" line, e.g. `cd <target_repo>; git apply <<'EOF' ...`
  - **Every claim in the rationale and root-cause description that references
    upstream behavior ("consistent with upstream", "upstream does X", "same as
    CUDA/MPS") MUST cite a specific `file:line`. If no such citation can be
    found, omit the claim and state only what was directly observed (error
    message, test output, or code read during triage). An unsubstantiated
    upstream comparison inflates reviewer confidence in a statement that was
    never verified.**
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
- **Never remove any pre-existing label that is not in the `agent:*`
  namespace.** This includes but is not limited to `issue_handler_handle`,
  `module: *`, `hw: *`, `dtype: *`, `os: *`, `bug`, `enhancement`,
  `type: *`, `skipped`, `ut_upstream`, `not_target`, `wait_upstream`,
  `dependency component: *`, `performance`, `E2E`, `Accuracy`. These
  labels are set by humans or by other automation (batch selectors,
  triage bots) and this skill has no authority to modify them.
  Explicitly: do NOT remove `issue_handler_handle` after processing an
  issue. It is a batch-selector label, not a "todo done" marker; the
  batch orchestrator uses it to know which issues are in scope, and
  removing it silently breaks the batch state. The skill's only allowed
  label operations are adding or removing labels in the `agent:*`
  namespace (`agent:active`, `agent:triaged`, `agent:done`,
  `agent:needs-human`, `agent:waiting-upstream`).
- **`agent:triaged` label requires a real `fix/root-cause` run.** Do not
  apply that label from a stage that only ran `fix/reproduce`.
- **`PATCH_PROPOSED` requires** Stage 4 -> Stage 5 PASSED -> Stage 5.5
  APPROVE. No speculative diffs.
- **Do not open a PR on any repo other than `pr_repo`.** If
  `target_repo != pr_repo`, use patch-proposal only. The default
  `pr_repo` is the repo that hosts the issue.
- **Do not commit the fix while `patch_proposal_mode` is in flight.**
  `fix/implement` leaves changes staged and Stage 6 reads them with
  `git diff --cached`; the single local audit commit made *after* Stage 6
  has posted the diff (see "Pipeline mode: per-issue branch isolation")
  is the only commit allowed, and it is never pushed.
- **Never run `fix/implement` from `main` or another issue's branch.**
  In pipeline mode with multiple issues, each issue's fix lives on
  `agent/issue-<N>`. If the branch is wrong, abort with `NEEDS_HUMAN`
  rather than commit onto the wrong branch.

## STRICT patch-acceptance rules

A patch is only `PATCH_PROPOSED` if it fixes the **root cause** of the
failure. This rule applies equally to the bug branch and to every
sub-bug produced in the skip-list branch. The Stage 5.5 reviewer MUST
reject anything that does not clear this bar.

**Acceptable root-cause fixes:**

- **(a) Correcting a stale test expectation** where upstream has
  legitimately changed observed behavior and the test itself is now
  wrong. Examples: exception class renamed (`UserError` → `Unsupported`),
  an error-message wording changed, or removing a dead `@skipIfXpu` on
  a test that genuinely passes now under its own assertion.
- **(b) Product-code fix** in `torch-xpu-ops` (kernel/op) or in
  `pytorch` runtime (framework code, dispatch, autograd, inductor,
  etc.). This includes fixing a test file that hard-codes CUDA APIs on
  a device-agnostic path, when the correct fix is to make the test
  device-agnostic (not to skip it on XPU).

**REJECTED — return `NEEDS_HUMAN`, not `PATCH_PROPOSED`:**

Any construct in the "skip-shaped workarounds" list in
`fix/implement` Step 3.5. That list is the authoritative catalogue
(new skip decorators, `DecorateInfo` skip entries, `raise SkipTest`,
loosening `atol`/`rtol` more than an order of magnitude without a
quantitative justification, hardcoded seeds, deleting the failing
assertion / test function, broad `try/except` suppressing the failure).
This orchestrator runs `fix/implement` with `allow_skip=false`, so
Step 3.5 will already have rejected these; Stage 5.5 is the
second-line defence.

Also rejected as "hide, don't fix":

- Any change whose stated rationale is "hide until real fix lands" /
  "unblock CI" / "align with MPS/CUDA which is also skipped here" —
  even if it does not literally match Step 3.5's syntactic patterns.

If reproduction shows the failure is a real product-code issue but the
root-cause fix is out of scope for this run (multi-day kernel work,
distributed hardware you do not have, a third-party component you can't
touch): outcome is `NEEDS_HUMAN` with the root cause and a clear
pointer to where the fix should live. That is an honest, valid outcome.
Prefer honest `NEEDS_HUMAN` over fabricated `PATCH_PROPOSED`.

The Stage 5.5 reviewer's checklist item 2 ("verify the changes fix the
root cause") IS this rule. If the reviewer is tempted to APPROVE a diff
that matches any bullet in the REJECTED list, that is a pipeline bug —
bail with `NEEDS_HUMAN`, name the real fix location, and post a state
comment saying so. Do not paper over.
