# Execution Modes (shared reference)

Shared contract for the `issue-handler` orchestrator and its leaf
skills (`issue-triage`, `fix-reproduce`, `fix-root-cause`,
`fix-implement`, `fix-verify`). Every skill that reports results or
touches a GitHub issue follows the mode rules below. Decide the mode
once at the start of a run and keep it for every stage.

## The two modes

- **Interactive mode (default).** The skill is loaded in a chat
  session with a human present. When a stage hits a blocker, a
  `NEEDS_HUMAN` verdict, an ambiguous classification, a failure it
  cannot reproduce, or a fix that will not verify, **ask the user
  and wait for input** instead of stopping silently. Report progress
  and results conversationally. Do **not** write status markers /
  labels into the GitHub issue body or leave comments unless the
  user explicitly asks.
- **Pipeline mode (explicit).** Selected only when the caller states
  the run is automated / non-interactive / "in the pipeline". No
  human to ask, so: write status into the issue body, advance the
  `agent:status` marker and labels, let leaf skills leave their
  own `<!-- agent:<name> -->` comments, and stop when the pipeline
  reaches a terminal verdict.

Whenever a skill says "leave a comment and stop" or "update the
issue body", that is the **pipeline-mode** action. The
**interactive-mode** equivalent is to surface the same information
to the user and ask how to proceed.

## Issue-body status contract (pipeline mode only)

In interactive mode, do not touch the issue body, markers, or labels
unless the user asks — report to the user instead.

In pipeline mode the `issue-handler` orchestrator does this
directly — no script. To stay compatible with any tooling that still
parses issue bodies, preserve the markers defined in the agent body
templates:

- Bug issues: `.github/ISSUE_TEMPLATE/agent/agent-issue-body.yml`
- Non-bug issues: `.github/ISSUE_TEMPLATE/agent/agent-issue-body-nonbug.yml`

When updating an issue body, keep these contracts intact:

### 1. Status marker

`<!-- agent:status:STAGE -->` at the top of the body. Advance
`STAGE` through the orchestrator's pipeline:

```
DISCOVERED → TRIAGING → REPRODUCING → TRIAGED →
IMPLEMENTING → VERIFYING → DONE
```

Terminal alternates:

- `SKIPPED` — non-bug (Stage 1), no longer reproduces (Stage 2),
  reproduction_missing (Stage 1).
- `NEEDS_HUMAN` — any leaf returned `NEEDS_HUMAN`, `CANNOT_VERIFY`,
  or attempts exhausted (Stage 5 loop bound).
- `DONE` — pipeline reached Stage 5 with `PASSED`.

### 2. Stage → label mapping

Apply the matching GitHub label when advancing the marker:

| Stage(s) | Label |
|----------|-------|
| DISCOVERED, TRIAGING, REPRODUCING, IMPLEMENTING, VERIFYING | `agent:active` |
| TRIAGED | `agent:triaged` |
| DONE, SKIPPED | `agent:done` |
| NEEDS_HUMAN | `agent:needs-human` |

The `agent:active` label sticks across all in-progress stages so
external filters (dashboards, CI monitors) can bucket "currently
being worked on" cleanly.

### 3. Action Items checklist

Check off `- [ ]` items as stages complete and fill the matching log
placeholders in the template.

**One comment per fix session (leaves produce text, the orchestrator
posts).** Leaf skills do NOT post to the issue — each returns its
`<!-- agent:<name> -->` report block on stdout (per each leaf's own
"the skill does not post comments" contract). The `issue-handler`
orchestrator owns **a single session comment**, edits it in place as each
stage completes, and never creates a second one:

- The invoking workflow may hand over a comment it already created, in
  `$SESSION_COMMENT_ID`. When that is set, adopt it as the session
  comment; otherwise create one at Stage 1.
- Stage 1 (triage) **fills** the comment: the `<!-- agent:session -->`
  marker followed by the triage block, replacing whatever placeholder
  text it held.
- Stages 3-5 (root-cause, implement, verify) each **append** their block
  to that same comment.
- Stage 6 **appends** the closing summary: verdict, branch, caveats.

Edit in place by id — read the current body, append, PATCH it back:

```bash
gh api "/repos/$OWNER/$REPO/issues/comments/$SESSION_COMMENT_ID" \
  --jq .body > session_comment.md
cat next_block.md >> session_comment.md
gh api --method PATCH "/repos/$OWNER/$REPO/issues/comments/$SESSION_COMMENT_ID" \
  -f body=@session_comment.md
```

Keep the accumulating body under 65000 chars (the GitHub limit is 65536,
and a PATCH over it fails outright): if the diff in the implement block
would blow past it, include the key hunks only.

The comment is public and editing does not retract anything (edit history
stays visible), so it carries stage reports only — never raw command
output, environment, or credentials.

The implement block shows the **diff** (`git diff --cached`, or the
key hunks), not a prose description of what changed — the analysis
already lives in the root-cause block above it.

For re-run detection (Stage 0), locate the single session comment by
its `<!-- agent:session -->` marker; the per-stage `<!-- agent:<name> -->`
markers are sub-headings within that one comment.

| Block within the session comment | Producing leaf skill |
|---|---|
| `<!-- agent:triage -->` | `issue-triage` |
| `<!-- agent:reproduce -->` | `fix-reproduce` (only on standalone `@torchxpubot reproduce`; not on pipeline runs) |
| `<!-- agent:root-cause -->` | `fix-root-cause` |
| `<!-- agent:implement -->` | `fix-implement` (diff, not prose) |
| `<!-- agent:verify -->` | `fix-verify` |
| `<!-- agent:summary -->` | `issue-handler` (Stage 6 closing summary) |
| `<!-- agent:batch-fanout -->` | `issue-handler` (batch fan-out summary, skip-list or heterogeneous) |

### 4. Canonical section headings

The `issue-triage` skill lays out the skeleton headings; their
content is filled across later stages. Bug issues use exactly:
`Description, Reproducer, Error Log, Environment, Test Info, Root
Cause Analysis, Proposed Fix Strategy, Target Repository,
Additional Context` — where `Root Cause Analysis`, `Proposed Fix
Strategy`, and `Target Repository` are filled at Stage 3 (root
cause), not at Stage 1. Non-bug issues use: `Description,
Objective, Current Status`.

## Per-stage ownership summary

- **Stage 1** (`issue-triage`) owns: the `<!-- agent:triage -->`
  report block (returned to the orchestrator, which fills the session
  comment with it); the `agent:status:DISCOVERED → TRIAGING`
  transition; section-heading skeleton.
- **Stage 2** (`fix-reproduce`) owns: the `refined_command`
  extracted for downstream stages; the
  `agent:status:REPRODUCING → TRIAGED` transition (via the
  orchestrator's reading of the reproduce verdict). Does NOT post a
  comment on issue-handler pipeline runs (comments are for
  standalone `@torchxpubot reproduce` invocations).
- **Stage 3** (`fix-root-cause`) owns: the `<!-- agent:root-cause -->`
  report block (returned to the orchestrator, which appends it to the
  session comment); `Root Cause Analysis`, `Proposed
  Fix Strategy`, `Target Repository` sections filled in the issue
  body.
- **Stage 4** (`fix-implement`) owns: the `<!-- agent:implement -->`
  report block — a **diff**, not prose — returned to the orchestrator,
  which appends it to the session comment.
- **Stage 5** (`fix-verify`) owns: the `<!-- agent:verify -->` block.
- **Stage 6** (the orchestrator's own wrap-up, not a leaf) owns: advancing
  `agent:status` to the terminal value; final Action Items check; the
  `<!-- agent:summary -->` closing block appended to the session comment;
  in batch fan-out runs (Stage 1u), the `<!-- agent:batch-fanout -->`
  summary comment.

## Reset-between-entries recipe (batched fan-out)

Both orchestrators run a fan-out loop over independent
entries (`issue-handler`'s Stage 1u batch path, `xpu-nightly-ci-fix`'s
nightly batch). Each entry is a separate sub-bug and can triage to a
different `target_repo`, so a prior entry's staged diff must not
bleed into the next. Both orchestrators use this identical recipe;
it lives here so the two copies cannot drift.

Capture the two independent base SHAs **once**, before entering the
loop:

```bash
pytorch_base=$(git -C $pytorch_dir rev-parse HEAD)
xpu_ops_base=$(git -C $pytorch_dir/third_party/torch-xpu-ops rev-parse HEAD)
```

Track them as two separate variables — a torch-xpu-ops fix bases off
`xpu_ops_base` while `pytorch_base` stays pinned for the pytorch tree,
and a pytorch fix does the reverse. Do not conflate them.

Reset **both** checkouts before each entry:

```bash
git -C $pytorch_dir reset --hard $pytorch_base
git -C $pytorch_dir clean -fdx
if [ -d "$pytorch_dir/third_party/torch-xpu-ops/.git" ]; then
    git -C $pytorch_dir/third_party/torch-xpu-ops reset --hard $xpu_ops_base
    git -C $pytorch_dir/third_party/torch-xpu-ops clean -fdx
fi
```

Loop bound inside a single entry (Stage 4 <-> Stage 5 retry):
**maximum 3 fix attempts**, matching the legacy pipeline's
`max_agent_attempts`. On attempts exhausted, record
`NEEDS_HUMAN(reason=attempts_exhausted)` for that entry and continue;
do not abort the whole loop on any single entry's failure.

## Re-run gate (pipeline mode)

`agent:active` issues get re-triggered often. The orchestrator's Stage 0
decides first-run vs re-run using two comment queries, shared here so the
"who is the bot" and "what counts as new" rules stay in one place:

- **Last agent comment timestamp** — the most recent comment whose body
  matches `<!-- agent:`. Empty → first run (run the full pipeline).
- **New human feedback** — any comment authored by a login other than
  the bot account (`$BOT_LOGIN`, the author of the `agent:*` comments)
  **and** created after the last agent comment. Its presence makes the
  run a **human-feedback re-run**: read the feedback verbatim and
  prepend it to the failure description handed to `fix-root-cause`
  (no new leaf input — the leaf already takes a free-form description),
  then run the full pipeline (no skip fast-paths). Human feedback
  outranks any cached verdict.

When there is no new human feedback, it is a **bare re-run**: run only
Stage 1 (triage) + Stage 2 (reproduce), then lean on `fix-root-cause`'s
`<!-- agent:root-cause -->` `analyzed_sha` fast-path — if reproduce is
identical and `target_repo` HEAD sha equals the recorded `analyzed_sha`,
the leaf re-emits the prior verdict and the orchestrator stops. Reproduce
differing, or the sha having moved, resumes the pipeline from Stage 3.
This is the only cross-run "skip redundant work" mechanism; it reuses the
leaf's existing sha check rather than adding orchestrator-side caching.
