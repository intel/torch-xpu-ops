# Case Assessment, Routing, and Ledgers

This reference owns semantic assessment, state vocabularies, filing derivation,
and coverage. Runtime facts come only from audited evidence.

## Assessment record

Write one `artifacts/assessments/<case_key>.json` per runtime-terminal case:

```json
{
  "case_key": "pytorch-pytorch-issue-123",
  "evidence_path": "artifacts/evidence/pytorch-pytorch-issue-123.json",
  "reproduction_status": "matched-upstream",
  "proof_gates": {
    "runtime": "pass",
    "bug": "pass",
    "xpu_fix": "pass",
    "canonical": "pass"
  },
  "oracle_audit": {
    "current_source": "<URL, test, docs, or diff>",
    "input_contract": "supported",
    "determinism": "repeatable",
    "replacement_checks": [],
    "verdict": "valid"
  },
  "issue_validity": "actionable-bug",
  "execution_path": "xpu-native",
  "execution_path_proof": "<dispatch, fallback, key-op, or code-path evidence>",
  "resolution_scope": "xpu-fix-required",
  "resolution_proof": "<code owner and why a shared fix is insufficient>",
  "behavior_canonical_url": "<URL or null>",
  "xpu_tracking_canonical_url": null,
  "tracker_origin": "none",
  "fix_pr_urls": [],
  "canonical_audit": {
    "tracking_searches": [
      {"query": "<query>", "retrieved_at": "<ISO-8601>", "result_urls": []}
    ],
    "tracker_state": null,
    "fix_pr_states": [],
    "verdict": "complete"
  },
  "tracking_repository": "intel/torch-xpu-ops",
  "implementation_repository": "intel/torch-xpu-ops",
  "final_verdict": "confirmed-xpu-issue",
  "filing_disposition": "file-xpu-tracker",
  "assessed_at": "<ISO-8601>",
  "live_state_refreshed_at": "<ISO-8601>",
  "assessment_status": "valid"
}
```

Runtime, bug, and XPU-fix gates are `pass`, `fail`, or `unresolved`; canonical is
`pass` or `unresolved`. `assessment_status` is `pending`, `valid`, or `invalid`;
it describes contract consistency, not whether the case is a bug. Set it to
`valid` only after evidence is valid, every field uses an allowed value, proof
citations resolve, and final fields match the ordered derivation below.

Derive gates; never assign them independently:

| Gate | `pass` | `fail` | `unresolved` |
|---|---|---|---|
| Runtime | `matched-upstream`, or `different-failure` with its own reference oracle; intended stage reached and fidelity valid | `not-reproduced` | `different-failure` without an independent oracle, `oracle-not-reached`, `blocked-script-error`, or `verification-gap` |
| Bug | `actionable-bug` or `input-validation-defect` | feature, design, invalid, or undefined/nondeterministic behavior | `verification-gap` |
| XPU fix | `xpu-fix-required` with `xpu-native` or `xpu-compiler` proof | `shared-fix-covers-xpu`, `upstream-design`, or `no-fix` | `unknown` path or scope |
| Canonical | `canonical_audit.verdict: complete`; searches and live tracker/fix-PR states, including absence, are recorded | not used | audit absent, incomplete, or stale |

For `oracle_audit`, use `input_contract: supported|invalid|unknown`,
`determinism: repeatable|not-repeatable|unknown`, and
`verdict: valid|invalid|unresolved`. Replacement checks and proof fields are
cited free text or structured records, not enums.

## Bug proof

Assign one `issue_validity`:

| Value | Meaning |
|---|---|
| `actionable-bug` | Supported deterministic behavior violates reference semantics. |
| `input-validation-defect` | Invalid input exposes missing or inconsistent validation that needs a fix. |
| `feature-request` | Requested behavior is outside the current contract. |
| `design-limitation` | Resolution requires an API, performance, or design decision. |
| `invalid-repro` | The repro changes the scenario or never tests its claim. |
| `undefined-or-nondeterministic` | The claim depends on uncontrolled data or behavior. |
| `verification-gap` | Available facts cannot decide validity. |

Reconstruct the oracle from current code, tests, docs, linked diffs, and live
discussion. Check input preconditions; replace uninitialized values, unseeded
randomness, non-Hermitian inputs, and other undefined inputs with deterministic
valid equivalents. Preserve the upstream scenario. Require two clean attempts
with the same decisive stage/signature before either actionable defect value.

## XPU-fix proof

Assign one `execution_path`:

- `xpu-native`
- `xpu-compiler`
- `shared-frontend`
- `cpu-fallback`
- `unknown`

Assign one `resolution_scope`:

- `xpu-fix-required`: XPU kernel, dispatch, lowering, or backend code needs an
  independent change, even when that code is in `pytorch/pytorch`.
- `shared-fix-covers-xpu`: one shared-core fix naturally corrects XPU.
- `upstream-design`: upstream design ownership must decide.
- `no-fix`: validity review leaves no defect.
- `unknown`: ownership or fix effect is unproved.

Compare CPU/reference and CUDA behavior when available, dispatch ownership,
affected files, and linked diffs. `shared-frontend` and `cpu-fallback` cannot
prove `xpu-fix-required`.

## Canonical and filing proof

Keep separate:

- `behavior_canonical_url`: general upstream behavior/design record
- `xpu_tracking_canonical_url`: canonical issue tracking the XPU-specific fix
- `tracker_origin`: `none`, `preexisting`, or `created-this-run`
- `fix_pr_urls`: open, merged, closed-unmerged, or superseded fixes
- `canonical_audit`: search queries/results, retrieval times, type-specific
  tracker and fix-PR states, and `verdict: complete|incomplete`

Search the configured tracking repository independently. An upstream behavior
issue does not replace an XPU tracker. Choose the implementation repository from
code ownership, independently of the tracking repository.

Apply the first matching row:

| Condition | `final_verdict` | `filing_disposition` |
|---|---|---|
| Runtime gate fails | `non-issue` | `no-issue` |
| Runtime passes and bug gate fails | `non-issue` | `no-issue` |
| Runtime/bug pass; XPU-fix fails with `shared-fix-covers-xpu` | `track-upstream` | `track-upstream` |
| Runtime/bug pass; XPU-fix fails with `upstream-design` or `no-fix` | `non-issue` | `no-issue` |
| Any gate required by the remaining rows is unresolved | `verification-gap` | `needs-evidence` |
| All gates pass, no tracker | `confirmed-xpu-issue` | `file-xpu-tracker` |
| All gates pass, preexisting tracker | `confirmed-xpu-issue` | `use-existing-xpu-tracker` |
| All gates pass, tracker created this run | `confirmed-xpu-issue` | `filed-xpu-tracker` |

Creating or finding a tracker never changes a proved XPU issue's final verdict.
An active competing XPU fix PR blocks handler implementation but not confirmation.
Refresh canonical and PR state immediately before every GitHub write and handoff.

## Ledgers and transitions

`candidate_ledger.jsonl` has one row per raw source:

- `id`, `kind`, `title`, `url`, selected timestamp/value
- `details_path`: null only for title reject
- `selection_status`: `title-reject`, `deep-reject`, or `selected`
- `selection_reason`
- `case_key`: required only for selected sources
- reclassification source, reason, and time when applicable

`case_ledger.jsonl` has one row per selected case:

- `case_key`, `source_ids`, `upstream_refs`
- `case_status`
- `state_history`: ordered `{from, to, reason, time, incident_id}` records
- `blocker_type` and structured blocker/incident reference when applicable
- `environment_incident`
- evidence/assessment paths and statuses
- reproduction status, verdict, canonical URLs, repositories, and filing disposition

`case_status` transitions only forward:

```text
repro-construction -> repro-ready
repro-construction -> blocked
repro-ready -> blocked
blocked -> repro-construction
blocked -> repro-ready
repro-ready -> runtime-terminal -> assessed
```

A case may start at `repro-construction` or `repro-ready`. Reclassification
requires newly discovered source facts plus prior state, reason, and time.
`evidence_status` and `assessment_status` are `pending`, `valid`, or `invalid`.
Every selected source appears in exactly one case row; rejected sources never
require a case row.

Start history with `from: null` and either allowed initial state. The current
`case_status` must equal the final history entry's `to`.

Use `blocker_type: construction`, `environment`, `security`, `fetch`, `platform`,
or `performance-harness`. A blocked case requires a structured blocker or
incident but no fabricated evidence or assessment. Coverage derives blocked keys
only from `case_status: blocked`; it derives pending keys from
`repro-construction`, `repro-ready`, or `runtime-terminal`.

After a remedy, mark the incident `retrying`, append a transition from `blocked`
to the recorded resume state, clear the active blocker, and rerun the case. A
successful clean retry resolves the incident; a failed retry reactivates it and
returns the case to `blocked`. Audit every adjacent history pair against the
graph.

## Coverage and authoritative status

`artifacts/coverage.json` records sorted raw and source-ledger id arrays, sorted
selected and case-ledger key arrays, stage/status/verdict counts, pending and
blocked case keys, active/retrying/resolved incident ids, `scan_audit_status`
(`pending`, `pass`, or `fail`), `review_status`, and `workflow_status`.

Derive status in this order:

1. `blocked` when any current external, active environment, security, fetch,
   platform, performance, or construction blocker prevents the next action.
2. `completed` when all completion conditions below hold.
3. `partial` when required work remains and can continue or retry now.

Completion requires:

- all collection shards completed and raw/source ids match
- selected/case keys match and every case is `assessed`
- all evidence and assessments are valid
- no active/retrying incident or pending/blocked case remains; resolved incidents
  may stay in history
- scan audit and independent review both equal `pass`
- coverage, scan report, review conclusions, and dashboard counts agree

Reports render this value and never define or override workflow status.
Any later filing, comment-driven reassessment, or live canonical-state change
that modifies an assessment or review manifest first moves a completed workflow
back to `partial`. Regenerate derived artifacts and repeat the full independent
review before returning to `completed`.

Candidate ledger owns source provenance/selection and case ledger owns membership
and `case_status`. Evidence and assessment files own their detailed statuses,
runtime result, verdict, canonical state, and filing disposition; corresponding
case-ledger fields are a materialized index only. Any mismatch fails audit, and
the index must be regenerated from its authoritative artifacts.
