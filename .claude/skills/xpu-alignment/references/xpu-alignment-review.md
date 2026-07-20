# Filed-Issue and PR Review

Review filed alignment objects after case assessment, or caller-supplied objects
directly. Runtime evidence and semantic assessment remain authoritative; review
refreshes public state and fix claims.

## Review rules

1. Confirm the object describes the assessed behavior and XPU execution path.
   Nearby failures, fallback, and environment incidents are not substitutes.
2. Refresh every issue and PR live. Distinguish `open`, `merged`,
   `closed-unmerged`, `superseded`, and `design-blocked`; closed is not fixed.
3. Re-evaluate `resolution_scope` from the proposed diff. A shared fix that
   covers XPU changes the case to `track-upstream`.
4. Keep tracking and implementation repositories separate. XPU-specific changes
   may belong in `pytorch/pytorch` while the tracking issue remains in the XPU
   repository.
5. Compare reference/CUDA behavior, dispatch ownership, and performance. Reject
   device-to-host synchronization in asynchronous hot paths unless the contract
   requires it.
6. Reject partial capability claims and fixes that only change the repro instead
   of the behavior.
7. Require proportional verification. Python changes need a clean tested build;
   C++/SYCL/ATen changes need a rebuilt artifact and targeted regression test.
8. Maintain one XPU tracking canonical and mark duplicate or superseded issues,
   branches, PRs, and comments.

Record `fix_state` separately from the case's `final_verdict`:

- `merged-upstream`: the fix landed but is not proved present locally.
- `verified-in-tested-build`: the tested environment contains the fix and the
  case repro passes.

Only `verified-in-tested-build` means the fix is verified. A missing rebuild,
stale wheel, or still-failing repro leaves `fix_state` at `merged-upstream` or
`verification-gap`; it does not add a `fixed` case verdict.

An open XPU-specific fix PR does not remove the tracking issue, but it blocks a
duplicate handler implementation. Refresh state again immediately before filing,
commenting, closing, or handoff.

## Review artifacts

Store one entry per object in `artifacts/review_state.json` with URL, type, live
state, labels, linked objects, retrieval time, and PR head/base/merge/review
state.

Write one section per object in `reports/review_conclusions.md`:

```md
## <n>. <repository>#<number> — <title>

- **Object / live state:** <type, URL, state>
- **Case assessment:** <path, final verdict, resolution scope>
- **Ownership:** <XPU backend | shared core | upstream design | environment>
- **Fix evidence:** <build, test, and remaining gaps>
- **Fix state:** <unfixed | merged-upstream | verified-in-tested-build | verification-gap>
- **Canonical outcome:** <tracking issue, behavior issue, and fix PRs>
- **Verdict:** <track | needs XPU fix | duplicate | needs design | verification-gap>
- **Required action:** <owner and next action>
- **Public action:** <none | draft id | posted URL>
```

After all object conclusions are written, synthesize
`reports/review_dashboard.md`. Derive every claim from the case assessment,
review state, and fix evidence. Include:

1. A title naming the review date or scan window and the exact reviewed object
   range/list, plus explicit exclusions and why they were excluded.
2. A per-object verdict table with object, short title, real bug status, fix
   owner/scope, whether an independent XPU fix is required, fix/PR quality,
   canonical outcome, and recommended action or label.
3. Bucket summaries grouping objects by actionable XPU fix, track-upstream,
   design-blocked, duplicate, invalid/non-issue, environment incident, and
   verification gap. Omit empty buckets.
4. The count and percentage of reviewed objects that require independent XPU
   code changes. Keep unresolved cases out of both numerator and confirmed count.
5. Systemic findings about alignment triage and handler output, but only when at
   least one reviewed object supplies concrete evidence. Cite the object ids.
6. Ordered priorities for close, track, investigate, rework, or implement, then
   a concise bottom line.

Keep `review_dashboard.md` distinct from `review_conclusions.md`: conclusions
are the auditable per-object record; the dashboard is the batch-level decision
summary. Do not call the batch complete when any in-scope object lacks a
conclusion; write `STATUS=partial` and list the missing objects instead.

Create `reports/review_comment_drafts.md` only for factual corrections,
ownership/fix-scope decisions, verification gaps, or duplicate/superseded
outcomes. Any claim about local XPU behavior cites the case evidence and
assessment. Generic "also reproduces on XPU" language is prohibited.

Summarize proposed comments, reviews, labels, and closures. Execute only the
exact actions separately authorized by the user, then record their URLs.
Publishing `review_dashboard.md` as a GitHub issue or comment is also a separate
write action and requires explicit authorization naming the target repository
and destination.
