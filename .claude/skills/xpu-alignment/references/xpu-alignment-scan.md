# Scan Execution

## Window and collection

Search source repository `pytorch/pytorch`. Interpret an inclusive date window
in UTC as
`[start 00:00:00Z, end + 1 day 00:00:00Z)`. Include:

- issues by `created_at`
- PRs by `created_at`
- default-branch commits by committer timestamp

Store the selected timestamp and value for every source. Search all states,
paginate, and split the half-open interval until every shard is below provider
result caps. If one timestamp still exceeds a search cap, switch to an uncapped
cursor API or partition deterministically by source type and numeric id/SHA;
never keep splitting an identical timestamp. Record shard query, bounds,
secondary partition, pages/cursors, source count, and completion at
`artifacts/collection_shards.json`. A capped or failed shard keeps the run
`partial` when it can retry now, otherwise `blocked`; never shrink or sample the
window.

Save source metadata in `artifacts/raw_candidates.json`. Use ids
`pytorch-pytorch-issue-<number>`, `pytorch-pytorch-pr-<number>`, and
`pytorch-pytorch-commit-<first-12-sha>`. Initialize exactly one
`candidate_ledger.jsonl` row per raw source.

## Selection and cases

Use `selection_status` from
[case assessment and ledgers](xpu-alignment-buckets-and-routing.md):

- A title reject has `details_path: null` and `case_key: null`.
- Fetch every non-title-rejected body, diff, linked change, and test into
  `artifacts/details/<id>.json`.
- A deep reject keeps its details path and has `case_key: null`.
- A selected source receives a case key and appears in exactly one case row.

Use an existing canonical issue id for `case_key` when available; otherwise use
the earliest stable PR id, then commit id. Preserve an assigned key when later
sources merge into the case. Merge only the same behavior/root cause or
issue/PR/commit chain; merging never deletes a source row.

Reject only clearly non-behavioral, documentation/CI/release-only, exact
duplicate, or platform-exclusive work with no shared/XPU path. Missing context
or a missing standalone repro is not a reject.

For every `repro-construction` case, record fetched inputs, attempted
constructions/adaptations, exact missing input, why no faithful oracle can yet
run, next action, owner, and timestamp. It remains pending or blocked and
prevents `workflow_status=completed`; a bare limitation label is invalid.

Retain a deterministic source-id-ordered sample for every reject reason.

## Construct and execute

Prefer the upstream repro or regression test. Preserve behavior, shapes, dtypes,
execution mode, and oracle; adapt only device mechanics. Write
`scripts/repro_<case_key>.py` with the terminal result required by the
[evidence contract](xpu-alignment-evidence-contract.md).

Execute serially with timeouts under the credential-free sandbox and per-attempt
isolation policy. A ready case stays ready after a failed attempt; repair the
repro or record the precise terminal blocker. Group shared failure signatures
into one environment incident and retry cleanly.

Run every potentially actionable bug in at least two independent process/cache
attempts before assessment. Do not reuse a process after a device assert or
crash.

## Coverage checkpoint

After collection and each stage, regenerate coverage from raw sources, both
ledgers, evidence, assessments, audit, and review state. Never hand-edit counts.
Only `artifacts/coverage.json.workflow_status` expresses full workflow status.
