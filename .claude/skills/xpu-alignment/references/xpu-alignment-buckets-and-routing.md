# Case Assessment, Routing, and Ledgers

This reference owns semantic assessment and filing derivation. Runtime facts come
from the evidence record; do not replace them with narrative judgment.

**Contents:** assessment record, bug proof, XPU-fix proof, canonical proof,
source/case ledgers, coverage.

## Assessment record

Write one `artifacts/assessments/<case_key>.json` per executed case:

```json
{
  "schema_version": 3,
  "case_key": "issue-123",
  "evidence_path": "artifacts/evidence/issue-123.json",
  "reproduction_status": "matched-upstream",
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
  "fix_pr_urls": [],
  "tracking_repository": "intel/torch-xpu-ops",
  "implementation_repository": "intel/torch-xpu-ops",
  "final_verdict": "confirmed-xpu-issue",
  "filing_disposition": "file-xpu-tracker",
  "assessed_at": "<ISO-8601>",
  "live_state_refreshed_at": "<ISO-8601>",
  "assessment_status": "PASS"
}
```

## Bug proof

Reconstruct the oracle from the current repro, API contract, issue body, linked
tests, fix diff, and live discussion. Do not trust a title, old report bucket, or
claimed exception class.

Assign `issue_validity`:

| Value | Meaning |
|---|---|
| `actionable-bug` | Supported deterministic behavior violates reference semantics. |
| `input-validation-defect` | An invalid input exposes missing or inconsistent validation that itself needs a fix. |
| `feature-request` | The requested behavior is not part of the current contract. |
| `design-limitation` | The behavior depends on an unresolved API/performance/design choice. |
| `invalid-repro` | The repro changes the upstream behavior or never tests its claim. |
| `undefined-or-nondeterministic` | The conclusion depends on undefined, uninitialized, or uncontrolled data. |
| `verification-gap` | Available facts cannot decide validity. |

Before accepting a bug:

- Check documented dtype, shape, stride, dimension, parameter, and device
  preconditions. Separate a normal validation error from a missing-validation
  defect.
- Replace `torch.empty`, uninitialized values, unseeded randomness, non-Hermitian
  linear-algebra inputs, and other undefined inputs with deterministic valid
  equivalents. If the behavior disappears, do not call the original functional
  claim a bug.
- Confirm repeatability in clean attempts. A changing exception, stage, or result
  is a verification gap.
- Preserve the exact upstream scenario. A nearby failure is not confirmation.

## XPU-fix proof

Assign one `execution_path`:

- `xpu-native`: the decisive operation ran in an XPU kernel/dispatch path.
- `xpu-compiler`: the decisive failure is in an XPU compiler/lowering path.
- `shared-frontend`: behavior ends before backend-specific execution.
- `cpu-fallback`: the decisive operation ran on CPU through fallback.
- `unknown`: available evidence cannot prove the path.

Then assign `resolution_scope`:

- `xpu-fix-required`: XPU kernel, dispatch, lowering, or backend code needs an
  independent change, even if that change belongs in `pytorch/pytorch`.
- `shared-fix-covers-xpu`: one shared/CUDA-driven core fix naturally corrects
  XPU without XPU-specific work.
- `upstream-design`: resolution depends on upstream API or design ownership.
- `no-fix`: no defect remains after validity review.
- `unknown`: code path or fix effect is not proved.

Compare CPU/reference behavior, CUDA behavior when available, dispatch ownership,
affected files, and linked fix diffs. `shared-frontend` or `cpu-fallback` alone
cannot prove `xpu-fix-required`. When ownership or fix effect is unknown, keep
`resolution_scope: unknown` and do not file.

## Canonical and filing proof

Keep three concepts separate:

- `behavior_canonical_url`: upstream issue/design record for the general behavior.
- `xpu_tracking_canonical_url`: existing issue tracking the required XPU fix.
- `fix_pr_urls`: live competing, merged, closed-unmerged, or superseded fixes.

An upstream behavior issue does not replace an XPU tracking issue when
`resolution_scope` is `xpu-fix-required`. Search the configured tracking
repository independently. Default to `intel/torch-xpu-ops`; use a different
repository only when the caller supplies an explicit experimental override.
Choose `implementation_repository` from the code owner, independently of the
tracking repository.

Derive the final fields:

| Condition | `final_verdict` | `filing_disposition` |
|---|---|---|
| `actionable-bug` or `input-validation-defect`, proved XPU-specific fix, no XPU tracker | `confirmed-xpu-issue` | `file-xpu-tracker` |
| Same valid defect, but XPU tracker already exists | `track-upstream` | `duplicate` |
| Valid defect, shared fix covers XPU | `track-upstream` | `track-upstream` |
| Feature/design/invalid/undefined/no-fix | `non-issue` | `no-issue` |
| Any proof gate unresolved | `verification-gap` | `needs-evidence` |

An existing XPU-specific fix PR does not eliminate the tracking issue requirement,
but it blocks duplicate implementation handoff. For a scanned case with an
existing XPU tracker, use `track-upstream`/`duplicate`; when reviewing that
canonical tracker itself, use the review object's verdict rather than changing
the case derivation.

Refresh all canonical and PR states immediately before filing. Live state wins
over a historical assessment: if a new shared fix supersedes an earlier
XPU-specific plan, change `resolution_scope`, record the prior scope and
reassessment reason, and derive the verdict again.

## Source and case ledgers

`candidate_ledger.jsonl` has exactly one row per raw source and contains only:

- `id`, `kind`, `title`, `url`
- `details_path`
- `selection_status`, `selection_reason`
- `case_key`
- reclassification audit fields when applicable

Selection is `title-reject`, `deep-reject`, `repro-construction`, or
`repro-ready`. Reclassification requires `reclassified_from`, a factual reason,
and time.

`case_ledger.jsonl` has exactly one row per `case_key` and contains:

- `case_key`, `source_ids`, `upstream_refs`
- `case_status`, structured construction limitation when present
- `environment_incident`
- `evidence_path`, `evidence_status`, `reproduction_status`
- `assessment_path`, `assessment_status`, `final_verdict`
- canonical URLs, repositories, and `filing_disposition`

Source rows never duplicate case evidence or verdict fields. Every `source_id`
appears in one case row, and every case source points to a source-ledger row.

## Coverage

`artifacts/coverage.json` uses schema version 3 and records raw/source-ledger id
counts and hashes, case-ledger key count and hash, selection counts, runtime
status counts, validity/scope/verdict counts, valid evidence and assessment
counts, pending case keys, and `run_status`.

`run_status` is `completed`, `partial`, or `blocked`. Completion requires exact
source equality, complete ready cases, valid cross-artifact references, and a
passing run audit. Partial runs may retain case-level PASS drafts but may not
claim full-window completion or emit a batch handoff.
