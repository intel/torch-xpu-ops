---
name: issue-handler
description: >
  Use when asked to fix a GitHub issue end-to-end, run the agent pipeline
  on an issue, or process an `agent:active` / batch tracking issue.
  Orchestrates the full pipeline: `issue-triage` → `fix-reproduce`
  → `fix-root-cause` → `fix-implement` → `fix-verify` → report. Handles
  single-bug issues and batch-bug issues (a parent tracking multiple
  sub-bugs — skip-list or heterogeneous — fanned out one fix branch per
  sub-bug). On re-run, prioritizes new human feedback and skips work
  that is still valid.
---

# Issue Handler — End-to-End Orchestrator

This is the **high-level scenario skill** for handling a single GitHub
issue. It does not do the detailed work itself; it sequences the leaf
skills into one iterative pipeline and reports the result. Each stage's
mechanics live in its own skill — read and follow that skill when you
reach its stage.

Every agent-produced diff is a **proposal**. This skill and the leaves
it calls never commit, push, tag, or open a PR — the invoking workflow
takes the staged diff after `fix-verify` passes and drives its own
PR-creation path with human review.

## Contents

- [Pipeline overview](#pipeline-overview)
- [Inputs](#inputs)
- [Execution modes](#execution-modes)
- Stage 0: [Re-run gate](#stage-0--re-run-gate-first-run-vs-re-run)
- Stage 1: [Triage](#stage-1--triage-issue-triage)
- Stage 2: [Reproduce](#stage-2--reproduce-fix-reproduce)
- [Stage 1u: Batch-issue fan-out](#stage-1u--batch-issue-fan-out) (skip-list + heterogeneous)
- Stage 3: [Root cause](#stage-3--root-cause-fix-root-cause)
- Stage 4: [Implement](#stage-4--implement-fix-implement)
- Stage 5: [Verify](#stage-5--verify-fix-verify)
- Stage 6: [Report](#stage-6--report)
- [Iterative loop bounds](#iterative-loop-bounds)
- [Issue-body status contract](#issue-body-status-contract)

## Pipeline overview

Single-bug path (default for `issue_type=single-bug`):

```
triage → reproduce → root-cause → implement → verify → report
                       ↑                          |
                       └─── loop up to 3 times ───┘
```

Batch path (`issue_type=batch-bug`, either `batch_kind`) — fan out the
single-bug pipeline per sub-item, each on its own fix branch:

```
triage → [preflight: install nightly wheel once]
       → for each sub-item:
             reset + reproduce
               ├─ NOT_REPRODUCED → stale (skip-list: mark STALE_SKIP + follow-up)
               └─ REPRODUCED → branch agent/fix-issue-<N>-<seq>-<slug>
                               → root-cause → implement → verify
             (any failure marks the sub-item and continues the batch)
       → fan-out report
```

skip-list vs heterogeneous differ only in how a `NOT_REPRODUCED`
sub-item is labeled; the loop is identical.

| Stage | Leaf skill | Purpose |
|-------|-----------|---------|
| 1. Triage | `issue-triage` | Text-only classification: single-bug / batch-bug (+ `batch_kind`) / nonbug, `scope`, `runtime_dependencies`, preliminary verdict |
| 2. Reproduce | `fix-reproduce` | Verify the failure still reproduces (three-stage fallback: nightly → source_build → ci_env) |
| 3. Root cause | `fix-root-cause` | Deep source analysis, `target_repo`, `domain`, `IMPLEMENTING`/`NEEDS_HUMAN` |
| 4. Implement | `fix-implement` | Edit code, stage the diff (never commit) |
| 5. Verify | `fix-verify` | Run the refined command against source build, PASSED/FAILED/CANNOT_VERIFY |
| 6. Report | this skill | Summarize outcome to the user (or into the issue in pipeline mode) |

## Inputs

- A GitHub issue on `intel/torch-xpu-ops` or `pytorch/pytorch` (URL,
  number, or raw body).
- `pytorch_dir` — path to a local pytorch checkout, resolved as
  described in `fix-reproduce` Prepare. If absent, this skill lets
  `fix-reproduce` / `fix-root-cause` clone it into
  `$XPU_OPS_ROOT/agent_space_xpu/pytorch/`.
- Mode (see below).

## Execution modes

The pipeline runs in one of two modes — **interactive (default)** or
**pipeline** — which changes how every stage reports results and
whether it writes to the GitHub issue. Decide the mode at the start
and pass it to every leaf. See
[references/execution-modes.md](references/execution-modes.md) for
the full contract.

- **Interactive (default):** ask the user when blocked; report
  conversationally; do not touch the issue body / labels / comments
  unless the user asks.
- **Pipeline (explicit):** no human to interrupt — advance the
  issue's `agent:status` marker, update stage labels, let leaf
  skills leave their `<!-- agent:<name> -->` comments, and stop when
  the pipeline settles on a terminal verdict.

## Stage 0 — Re-run gate (first-run vs re-run)

**Pipeline mode only; skip in interactive mode** (a human is already
driving, so just run the full pipeline). An `agent:active` issue is
frequently re-triggered — a maintainer leaves feedback, or the bot is
re-invoked with no new information. This gate decides, before spending
a build, whether this is a fresh run, a **human-feedback re-run**
(highest priority), or a **bare re-run** (skip work that already ran and
is still valid).

### Step 0.1: Detect prior agent activity

Find the most recent agent comment and its timestamp:

```bash
last_agent_ts=$(gh issue view "$N" --repo "$OWNER/$REPO" --json comments \
  --jq '[.comments[] | select(.body | test("<!-- agent:"))] | last | .createdAt // ""')
```

Empty `last_agent_ts` → **first run**. Skip the rest of Stage 0 and go
to Stage 1 normally.

### Step 0.2: Detect new human feedback

A comment is *human feedback* when it is authored by a non-bot account
**and** created after `last_agent_ts`. (The bot account is the one that
authored the `<!-- agent:* -->` comments; exclude it by login.)

```bash
new_human=$(gh issue view "$N" --repo "$OWNER/$REPO" --json comments \
  --jq --arg ts "$last_agent_ts" --arg bot "$BOT_LOGIN" \
  '[.comments[] | select(.author.login != $bot and .createdAt > $ts)] | length')
```

- `new_human > 0` → **human-feedback re-run**. Human feedback is the
  **highest priority signal.** Read every such comment verbatim and
  **prepend it to the failure description** you hand to Stage 3
  (`fix-root-cause` takes a free-form failure description; a leading
  "Maintainer feedback since last run: ..." block steers the
  re-analysis without any new leaf input). Run the **full** pipeline
  from Stage 1; do not take any of the skip fast-paths below. A human
  saying "still wrong" or "change X" overrides any cached verdict.
- `new_human == 0` → **bare re-run**. Continue to Step 0.3.

### Step 0.3: Bare re-run — skip what is still valid

No human pointed anything out since the last agent comment, so re-doing
the whole pipeline would just repeat identical work. Re-run **only** the
cheap front of the pipeline and compare against last time:

1. Run **Stage 1 (triage)** and **Stage 2 (reproduce)** as normal.
2. Compare the reproduce result to the previous run. The previous
   `refined_command` + verdict are recoverable from the last
   `<!-- agent:root-cause -->` comment's `analyzed_sha` context, or
   re-derived by reading the prior `<!-- agent:reproduce -->` /
   sweep comment. "Identical" means same verdict **and** same
   `refined_command`.
   - **Reproduce differs** (now passes, or a different command
     reproduces) → the situation changed on its own; resume the full
     pipeline **from Stage 3 (root-cause)** with the new reproduce
     result. Do not reuse the cached root-cause.
   - **Reproduce identical** → nothing observable changed. Hand off to
     Stage 3, which runs `fix-root-cause`'s own
     `<!-- agent:root-cause -->` `analyzed_sha` fast-path: if
     `target_repo` HEAD sha equals the recorded `analyzed_sha`, that
     leaf re-emits the prior verdict verbatim and this orchestrator
     **stops** (the earlier outcome — fix already staged, or
     `NEEDS_HUMAN` — still stands; there is nothing new to do). If the
     sha moved, `fix-root-cause` re-analyzes and the pipeline continues
     from Stage 3 as usual.

This never skips Stage 1/Stage 2 — they are cheap (text + nightly wheel)
and are the only way to notice the failure went away. It only avoids the
expensive Stage 3-5 rebuild+fix when both the observable failure and the
analyzed code are unchanged.

## Stage 1 — Triage (`issue-triage`)

Call `issue-triage` on the issue body + comments. It emits
`issue_type` (`single-bug` / `batch-bug` / `nonbug`), `batch_kind`
(`skip-list` / `heterogeneous` / `null`), `reproduction_missing`
(`yes` / `no`), `scope`, `runtime_dependencies`, and a preliminary
`handling` (`agent-fixable` / `needs-human`).

Branch on `issue_type` (triage already made the batch-vs-nonbug call;
no re-detection here):

- `nonbug` → stop the fix pipeline; skip to Stage 6 Report with
  `SKIPPED(reason=nonbug)`.
- `single-bug` with `reproduction_missing=yes` → stop; Stage 6 Report
  with `NEEDS_HUMAN(reason=reproduction_missing)`. `issue-triage`'s
  own comment already asks the reporter for a reproducer.
- `single-bug` with `reproduction_missing=no` → continue to Stage 2.
- `batch-bug` → **Stage 1u**, passing through `batch_kind` (the loop
  uses it only to label a `NOT_REPRODUCED` sub-item).



## Stage 2 — Reproduce (`fix-reproduce`)

Only for the single-bug path (`issue_type=single-bug`). Call
`fix-reproduce` with:

- `reproducer_command` — extracted by `issue-triage` from the issue
  body.
- `stage=auto` — full three-stage fallback.
- `ci_repo` — inferred from repo (`torch-xpu-ops` for issues on
  intel/torch-xpu-ops, `pytorch` for pytorch/pytorch), or the value
  the bot passes explicitly.

Branch on its verdict:

- `REPRODUCED` → continue to Stage 3. Record the `refined_command`
  and `base` for downstream stages.
- `NOT_REPRODUCED` → the issue is stale; Stage 6 Report with
  `SKIPPED(reason=no_longer_reproduces)`.
- `NO_REPRODUCER` → Stage 6 Report with
  `NEEDS_HUMAN(reason=no_reproducer)`.
- `CANNOT_VERIFY` → Stage 6 Report with
  `NEEDS_HUMAN(reason=cannot_verify)` and the `blocker` field.

## Stage 1u — Batch-issue fan-out

One loop for every parent issue that tracks multiple children. The
parent lists several sub-items; run the single-bug pipeline on each
**independently**, each on its own fix branch, and report every outcome
back on the parent. "Fix what you can" — a sub-item that can't be fixed
is marked and the batch continues.

Entered for `issue_type=batch-bug`. `issue-triage` already set
`batch_kind`; the two kinds share the entire loop and differ only in how
a **`NOT_REPRODUCED`** sub-item is labeled (see step 2 below):

- `heterogeneous` — the parent body lists *distinct* sub-bugs.
- `skip-list` — a `Bug Skip` issue listing *homogeneous* already-skipped
  tests. A `NOT_REPRODUCED` entry here means the skip decorator is now
  stale.

No child GitHub issues are created. Each sub-item is a checklist entry
on the parent; its fix lives on a dedicated branch so a human can open
one PR per fixed sub-item.

### Extract sub-items

Parse the parent body into a list of sub-items. Each is either:

- an inline sub-bug — a checklist line naming a test node id or
  reproducer (skip-list entries are always this form; normalize a bare
  `Class::method` to a node id, `fix-reproduce`'s Prepare step resolves
  the file), or
- a linked child reference `owner/repo#N` — fetch that issue and use
  its body as the sub-item's failure description. Only follow
  `intel/torch-xpu-ops` and `pytorch/pytorch` references; ignore any
  other repo (untrusted, per `fix-root-cause`).

Give each sub-item two identifiers used for branch naming:

- **`seq`** — 1-based position in the body checklist order. Readable and
  maps a branch back to its body line. Not the stable identity (editing
  or reordering the body changes it).
- **`slug`** — the stable identity, derived from the sub-item's test node
  id: take the leaf method name, strip device suffixes (`_cpu` / `_xpu` /
  `_cuda` / `_meta`) and any dtype suffix, lowercase, keep `[a-z0-9._-]`,
  truncate to 40 chars. If two sub-items produce the same slug, append
  `-2`, `-3`, … in body order so every slug is unique within the issue.

Branch name is `agent/fix-issue-<N>-<seq>-<slug>` (N = parent issue
number). On a **re-run**, match a sub-item to its prior branch by slug —
look for an existing `agent/fix-issue-<N>-*-<slug>` (any seq); if found,
that is the same sub-item (rename to the current seq if it moved, never
create a duplicate). Skip headers, prose, and empty lines.

### Preflight (many entries): install nightly wheel once

When the list is long (skip-list issues routinely have dozens),
front-load the one real wheel install so the per-entry
`fix-reproduce(stage=nightly)` calls each find the env already current
and return fast. `fix-reproduce` always issues `pip install --pre
--upgrade` (it refuses to trust a stale wheel); running it once here
does the real work. There is no `skip_wheel_install` flag — this is
purely an ordering optimization:

```bash
pip3 install --pre --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/xpu
python -c "import torch; print('nightly:', torch.__version__)"
```

### Per-sub-item loop

Capture the two base SHAs once, then for each sub-item reset both
checkouts per the shared
[reset-between-entries recipe](references/execution-modes.md#reset-between-entries-recipe-batched-fan-out)
— a prior sub-item's staged diff must not bleed into the next.

For each sub-item:

1. **Reset** both checkouts to the base SHAs (shared recipe).
2. **Reproduce** (`fix-reproduce`, `stage=auto`). It detaches the
   pytorch tree to its base (`origin/main` or the `ci_commit`
   fallback) and returns that `base` plus a `refined_command`. Branch
   on the verdict:
   - `REPRODUCED` → continue to step 3; keep its `base` and
     `refined_command`.
   - `NOT_REPRODUCED` → **stale**: nothing to fix. Record it and go to
     the next sub-item. If `batch_kind=skip-list`, additionally mark it
     `STALE_SKIP` and record a **follow-up** to remove the now-obsolete
     skip decorator — the orchestrator does **not** delete it here (see
     "Stale skips" below).
   - `NO_REPRODUCER` → **INVALID_ENTRY** (renamed/removed test, or
     malformed). Record, continue.
   - `CANNOT_VERIFY` → **UNVERIFIED** (environmental). Record, continue.
3. **Root-cause, then branch.** Run **Stage 3** (`fix-root-cause`)
   first — it returns `target_repo`, which decides *which* checkout the
   fix (and thus the branch) lives in:
   - `target_repo=pytorch` → `target_repo_dir = pytorch_dir`, base is
     the reproduce `base` on the pytorch tree.
   - `target_repo=torch-xpu-ops` →
     `target_repo_dir = pytorch_dir/third_party/torch-xpu-ops`, base is
     `xpu_ops_base` (the submodule's pinned commit from the shared
     recipe). `fix-implement` / `fix-verify` handle the `xpu.txt` pin
     rewrite internally; the orchestrator only creates the branch.

   Create the isolated fix branch on `target_repo_dir` off its base so
   the diff is pushable on its own:

   ```bash
   git -C "$target_repo_dir" checkout -B "agent/fix-issue-${N}-${seq}-${slug}" "$base"
   ```

   Then run **Stage 4 → 5** (same contract and 3-attempt bound as the
   single-bug path). `fix-implement` expects exactly this: a fresh
   branch the orchestrator just created, clean worktree.
4. **On any leaf `NEEDS_HUMAN` / `CANNOT_VERIFY` / `FAILED`** (or
   attempts exhausted): mark **this sub-item** blocked with the reason,
   record it, and **continue to the next sub-item** — never abort the
   whole batch on one hard sub-item.
5. **On `fix-verify` PASSED**: the staged diff sits on
   `agent/fix-issue-${N}-${seq}-${slug}`. Leave it staged for the
   invoking workflow to commit + push that branch and open one PR (with
   human review). Record the sub-item as fixed with its branch name, and
   write its `fix_result-${slug}.json` (see "Machine-readable outputs"
   below) so the bot's reviewer/gate can re-verify it per sub-item.

Leaf skills post their own `<!-- agent:root-cause -->` /
`<!-- agent:implement -->` / `<!-- agent:verify -->` comments per
sub-item on the **parent** issue (pass the parent issue number
through).

### Stale skips (`batch_kind=skip-list` only)

A `STALE_SKIP` sub-item's skip decorator can be removed, but this
orchestrator does **not** delete it. Deleting a skip is itself a code
change that needs its own verify + PR; folding it into this sweep would
mix "the skip is obsolete" with "and here's the removal diff" and
obscure the batch outcome. Instead, surface every `STALE_SKIP` in the
report as an explicit follow-up (candidate for a human, or a separately
invoked `fix-implement` run to remove the decorator). The report is the
hand-off; nothing is auto-removed.

### Fan-out report

Post one summary comment on the parent (or surface to the user in
interactive mode), and mirror it into the parent's checklist:

```
<!-- agent:batch-fanout -->

## Batch fan-out results

Base: <torch nightly version or base sha>

| Sub-item | Outcome | Branch / Reason |
|---|---|---|
| test_bar_xpu_float32 | FIXED | agent/fix-issue-4321-1-test_bar |
| test_baz | NEEDS_HUMAN | cross_repo_coordinated |
| test_qux | NEEDS_HUMAN | attempts_exhausted |
| test_old | STALE_SKIP | follow-up: remove skip decorator |
| test_gone | INVALID_ENTRY | does not collect |

- **FIXED:** N sub-items — one branch each, ready for a human to open
  a PR.
- **NEEDS_HUMAN:** M sub-items — see per-sub-item reason.
- **STALE_SKIP:** K sub-items (skip-list only) — no longer reproduce;
  follow up to remove the obsolete skip decorator.
- **INVALID_ENTRY / UNVERIFIED:** P sub-items — malformed/renamed, or
  environmental during reproduce.

*Automated by issue-handler.*
```

Omit category rows that have no members (a `heterogeneous` batch never
has `STALE_SKIP`). The `<!-- agent:batch-fanout -->` marker lets a
re-run locate and update this same comment in place. On a re-run
(Stage 0), only re-process sub-items that are not already `FIXED` on a
live branch, unless human feedback (Stage 0.2) reopens a specific one.

After the loop, go to Stage 6 Report with the aggregate outcome
(`IMPLEMENTING(fix_verified)` if any sub-item was fixed;
`NEEDS_HUMAN` only if *every* actionable sub-item needed a human —
`STALE_SKIP` follow-ups do not by themselves force `NEEDS_HUMAN`).

### Machine-readable outputs (pipeline mode)

The `<!-- agent:batch-fanout -->` comment is for humans. In pipeline mode
also write machine-readable files under `$AGENT_SPACE` (the gitignored
scratch dir) so the invoking bot workflow can drive per-sub-item
re-verification and PR creation without re-parsing the comment:

- **`batch_summary.json`** — one file listing every sub-item:

  ```json
  {
    "issue": 4321,
    "kind": "batch-bug",
    "batch_kind": "heterogeneous",
    "sub_items": [
      { "seq": 1, "slug": "test_bar", "outcome": "FIXED",
        "branch": "agent/fix-issue-4321-1-test_bar",
        "target_repo": "torch-xpu-ops",
        "fix_result": "fix_result-test_bar.json",
        "summary": "one-line what/why" },
      { "seq": 2, "slug": "test_baz", "outcome": "NEEDS_HUMAN",
        "branch": null, "reason": "cross_repo_coordinated" },
      { "seq": 3, "slug": "test_old", "outcome": "STALE_SKIP",
        "branch": null, "reason": "follow-up: remove skip decorator" }
    ]
  }
  ```

- **`fix_result-<slug>.json`** — for each `FIXED` sub-item, the same
  schema the single-bug fix job writes as `fix_result.json` (`needs_build`,
  `build_ok`, `xpu_available`, `pytorch_dir`, `refined_command`, `notes`),
  suffixed by slug so the bot's reviewer/gate loop can re-verify each
  sub-item independently.

Branch enumeration (`agent/fix-issue-<N>-*`) remains the bot's primary
source of truth; `batch_summary.json` only enriches it (target_repo,
summary line). Its absence is not fatal to PR creation.

## Stage 3 — Root cause (`fix-root-cause`)

Called on the single-bug path, and once per sub-item from Stage 1u.
Call `fix-root-cause` with the failure description and the
`refined_command` from Stage 2.

Branch on its `verdict`:

- `IMPLEMENTING(reason=ok)` → continue to Stage 4. Record
  `target_repo`, `domain`, `analyzed_sha`, `root_cause`,
  `fix_strategy`.
- `NEEDS_HUMAN` → Stage 6 Report with the specific
  `reason` (`task_or_feature` / `feature_gap` / `hardware_specific` /
  `cross_repo_coordinated` / `no_registered_domain` / etc.). Each
  reason maps to a different final `agent:status` value; see
  [execution-modes.md](references/execution-modes.md).

## Stage 4 — Implement (`fix-implement`)

Call `fix-implement` with `triage_result`, `pytorch_dir`,
`target_repo_dir` (derived from `target_repo`), and `allow_skip`:

- `allow_skip=false` for the standard issue-handler pipeline —
  never add skip decorators, must actually fix.
- `allow_skip=true` only when the caller explicitly opts in
  (e.g. `xpu-nightly-ci-fix` orchestrator with a nightly-CI issue).

Branch on the verdict:

- `READY(reason=ok)` → continue to Stage 5.
- `NEEDS_HUMAN` → Stage 6 Report. The specific `reason`
  (`skip_outside_target_repo` / `skip_guard_rejected` /
  `no_fix_possible` / etc.) drives the final label.

## Stage 5 — Verify (`fix-verify`)

Call `fix-verify` with `refined_command` (from Stage 2),
`target_repo_dir`, and `changed_files` (from Stage 4). `fix-verify`
unconditionally produces the FAIL->PASS before/after table and runs
`spin fixlint` on a passing result — no flags to pass.

Branch on the verdict:

- `PASSED(reason=ok)` → Stage 6 Report with
  `IMPLEMENTING(fix_verified)`. The staged diff is ready for the
  workflow to open a PR (with human review).
- `FAILED` → **loop back to Stage 4** with the failure output as
  additional context. See "Iterative loop bounds" below.
- `CANNOT_VERIFY` → Stage 6 Report with
  `NEEDS_HUMAN(reason=<verify's reason>)` and the blocker. Do not
  loop on CANNOT_VERIFY — the environment problem will not fix
  itself.

## Stage 6 — Report

At the end, summarize the outcome. In **interactive mode** present
this to the user in plain language. In **pipeline mode** advance
the issue's `agent:status` to the terminal stage
(`DONE` / `NEEDS_HUMAN` / `SKIPPED`) and update the checklist per
[execution-modes.md](references/execution-modes.md); the leaf skills
already left their own `<!-- agent:<name> -->` comments so no extra
summary comment is needed unless the pipeline is a batch fan-out (which
posts the `<!-- agent:batch-fanout -->` summary).

Always include in the summary:

- **Issue:** link/number and one-line title.
- **Path:** `single-bug` / `batch` (with `batch_kind`).
- **Outcome:** `IMPLEMENTING(fix_verified)` / `NEEDS_HUMAN(<reason>)`
  / `SKIPPED(<reason>)`.
- **Root cause** (from Stage 3, if reached).
- **Files changed** (from Stage 4, if reached).
- **Verification** (from Stage 5, if reached).
- For a batch: per-sub-item outcome + branch name (FIXED) or reason
  (NEEDS_HUMAN / INVALID_ENTRY / UNVERIFIED), plus any `STALE_SKIP`
  follow-ups when `batch_kind=skip-list`.

If the outcome is `IMPLEMENTING(fix_verified)`, the invoking
workflow reads the staged diff (`git -C $target_repo_dir diff
--cached`) and drives its own PR-creation path. **Do not open the
PR from this skill.**

## Iterative loop bounds

The pipeline is not strictly linear. Loop when a later stage
invalidates an earlier assumption:

- Stage 5 `FAILED` → return to Stage 4 (refine the fix).
- Stage 4's Step 3.5 skip-guard rejects → the leaf itself re-runs
  once; a second rejection returns `NEEDS_HUMAN` and the
  orchestrator does not retry.

Bound: **maximum 3 fix attempts** (Stage 4 → Stage 5 → Stage 4 …).
This matches the legacy pipeline's `max_agent_attempts`. When you
stop without success, report `NEEDS_HUMAN(reason=attempts_exhausted)`
with the last `fix-verify` failure output in `reason_detail`.

Do **not** loop on:

- `CANNOT_VERIFY` at any stage (environment problem, not fix
  problem).
- `NEEDS_HUMAN` from any leaf (contract: the leaf already decided
  it needs a human).
- Stage 3 `no_registered_domain` (domain registry is a fixed set,
  looping won't unstick it).

## Issue-body status contract

**Pipeline mode only.** In interactive mode, do not touch the issue
body/markers/labels unless the user asks — report to the user
instead.

This orchestrator owns advancing the overall `<!-- agent:status:X -->`
marker through:

```
DISCOVERED → TRIAGING → REPRODUCING → TRIAGED → IMPLEMENTING →
VERIFYING → DONE
```

with terminal alternates `NEEDS_HUMAN` and `SKIPPED`. Stage-by-stage
mapping to labels is in
[references/execution-modes.md](references/execution-modes.md); each
leaf skill owns its own `<!-- agent:<name> -->` comment/log slot,
this orchestrator owns the overall `agent:status` marker + the
Action Items checklist.
